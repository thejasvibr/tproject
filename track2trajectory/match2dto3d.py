'''
2D track to 3D trajectory matching
==================================
This module contains functions which perform multi-camera 2D track matching 
to their corresponding 3D trajectories.
'''
import cv2
import math
import numpy as np 
import pandas as pd
from scipy.spatial import distance
import track2trajectory.camera as camera
from tqdm import tqdm

def estimate_3d_points(camera1, camera2, df_2d: pd, df_3d_gt, result_file_name, fm_1,
                               **kwargs):
    '''
    Receives 2D projections from 2 cameras. Does epipolar matching, and after 10 
    frames applies Kalman Filtering, and gets robust pairing. 

    Does not assume that camera 1 and camera 2 have the same parameters. 

    Parameters
    ----------
    camera1 : camera.Camera
    camera2 : camera.Camera
    df_2d : pd.DataFrame
        2D projections of objects
    df_3d_gt : pd.DataFrame
        Groundtruth 3D data. For comparison of the matched trajectories and 
        the groundtruth.
    result_file_name : str
        Final file name that the results will be saved in. 
    fm_1 : np.array
        Fundamental matrix for camera 1
    kf_distance_threshold : float, optional
        Max allowable distance between Kalman Filter prediction. Defaults to
        0.3 m

    Returns 
    -------
    avg_ee : float
        Average estimation error in metres
    -1 : int
        Significance of the value unclear (says Giray)

    Side effects
    ------------
    Write 'result_files/{result_file_name}.csv' with columns:
        frame : frame number
        oid_1, oid_2 : object id in camera 1 and camera 2
        x,y,z : estimated xyz position
        dt1 : distance between nearest point and Camera 1 epipolar line projected onto Camera 2 in pixels
        dt2 : nearest point to epipolar line back projected onto camera 1 from the point that gave dt1 in pixels (the 'two-way authenticated point')
        dt3 : distance between Kalman fileter prediction for that point and the estimated xyz in metres.
              The value is -1 if there is not enough data to predict.
        gtx, gty, gtz : ground truth xyz coordinates
        ee : estimation error in metres for each point and frame 
        kcounter : kalman filter counter. Counts how many frames of data were available 

    TODO
    ----
    * During pairing, it's important to write a separate function which 
    checks that object ids are consistently matched - ideally after the recon2 output is created
    * Convert the epipolar line threshold to a user input variable instead of the currently hard-coded 50 pixels
    * Currently scount assumes object IDs are the same across the cameras -- needs to be generalised
    * np.isclose is used to check if two points are the same, and a hard-coded threshold of 0.0001 is used. This 
        may cause issues when data with high-precision tracking accuracy.
    * Right now All 2D points are assumed to be UNDISTORTED! Undistortion is applied however if distortion coefficients
    are provided. This needs to be checked once more for correctness.

    References 
    ----------
    * Giray's Master Thesis
    '''
    kf_distance_threshold = kwargs.get('kf_distance_threshold', 0.3) # metres
    
    df_2d = df_2d.sort_values('frame')
    df_t = df_2d.loc[df_2d['cid'] == camera1.id] # trajectories cam 1
    df_t_2 = df_2d.loc[df_2d['cid'] == camera2.id] # trajectores cam 2

    run_len = len(df_t)  # Number of frames.

    # Benchmark values
    # see Sections 3.2-3.3 in Giray's thesis for more background
    df_recon = pd.DataFrame(columns=["frame", "gtoid", "oid_1", "oid_2", "x", "y", "z", "dt1", "dt2", "dt3"])
    scounter = 0 # successful pairings
    fcounter = 0 # failed pairings  
    ccount = 0 # Sum total of all points across all frames in Camera 1
    avg_ee = 0 

    outfm = fm_1
    # _, _, pm2, pm3, _ = cv2.stereoRectify(camera1.cm_mtrx, camera1.cof_mtrx, camera2.cm_mtrx, camera2.cof_mtrx,
    # np.size(1920, 1080), camera1.r_mtrx, camera1.t_mtrx)

    for i in tqdm.trange(run_len):
        temp_p = np.float32([df_t.iloc[i]['x'], df_t.iloc[i]['y']])
        # temp_p = df_t.iloc[i:i+1]
        # Get proper frame values for comparison.
        fn = df_t.iloc[i]['frame']
        at_frame_c2 = df_t_2.loc[df_t_2['frame'] == fn]

        at_frame_c1 = df_t.loc[df_t['frame'] == fn]

        call_kalman_fill = False

        if (scounter + fcounter) != 0:
            print("Matching perf: " + str(scounter / (scounter + fcounter)) + " Scounter = " +
                  str(scounter) + " Fcounter = " + str(fcounter) +
                  " Matching rate: " + str(((scounter + fcounter) / i) * 100.0))
            call_kalman_fill = False
        if not at_frame_c2.empty and not at_frame_c1.empty:

            cont = True # continue flag - becomes False after correspondence checks with all 
            # available pionts in frames are done
            temp_c2 = at_frame_c2.copy() # all points on camera 2 at this frame

            # p1 = np.float64([cand1[0][0][0], cand1[0][0][1], 1])
            # p2 = np.float64([cand2[0][0][0], cand2[0][0][1], 1])
            # res = np.matmul(outfm[0], p1)
            # epilines = cv2.computeCorrespondEpilines(cand2, 2, outfm[0])
            ignore_candidate = False
            candidate_id_to_ignore = 0
            while cont:

                if ignore_candidate:
                    if len(temp_c2) > 1:
                        temp_c2 = temp_c2.drop(
                            temp_c2.loc[(np.isclose(temp_c2['oid'], candidate_id_to_ignore, 0.0001))].index)
                    ignore_candidate = False
                dist_2_points = -1
                cand_p_1, pre_min_1, pre_frame_1, pre_id_1, row_num1 = find_candidate(temp_c2, outfm, camera2,
                                                                                      camera1, temp_p, 20, 2000, False)

                if len(cand_p_1) != 0:
                    temp_c1 = at_frame_c1.copy()

                    cp1 = np.float32([cand_p_1[0], cand_p_1[1]])
                    # the 'two-way authentication' is done here. The epipolar of the 'best point' in Cam 2 is 
                    # is the projected onto Cam 1 - and the point closest to this epipolar is checked.
                    cand_p_2, pre_min_2, pre_frame_2, pre_id_2, row_num1 = find_candidate(temp_c1, outfm, camera1,
                                                                                          camera2, cp1, 20, 2000, False)

                    if len(cand_p_2) != 0:
                        # check that the '2-way auth' point on Cam1 is the same as the original 
                        # candidate point
                        if (np.isclose(temp_p[0], cand_p_2[0], atol=0.0001) and
                                np.isclose(temp_p[1], cand_p_2[1], atol=0.0001)):
                            # print("Successful match")
                            
                            candp2 = cv2.undistortPoints(cand_p_2, camera1.i_mtrx, camera1.cof_mtrx,
                                                         P=camera1.i_mtrx)
                            candp1 = cv2.undistortPoints(cand_p_1, camera2.i_mtrx, camera2.cof_mtrx,
                                                         P=camera2.i_mtrx)
                            tri_res = cv2.triangulatePoints(camera1.cm_mtrx, camera2.cm_mtrx,
                                                            candp2, candp1)
                            x_2 = tri_res[0] / tri_res[3]
                            y_2 = tri_res[1] / tri_res[3]
                            z_2 = tri_res[2] / tri_res[3]
                            # Get groun truth 3D value.
                            oid_f = df_t.iloc[i]['oid']
                            t1 = df_t.iloc[i]['frame']
                            t2 = df_3d_gt.loc[np.float32(df_3d_gt['frame']) == np.float32(t1)]
                            t4 = t2.loc[np.float32(t2['oid']) == oid_f]

                            xf = x_2
                            yf = y_2
                            zf = z_2
                            s_point = [x_2, y_2, z_2] # estimated xyz position

                            gtx = t4.iloc[0]['x']
                            gty = t4.iloc[0]['y']
                            gtz = t4.iloc[0]['z']

                            gt_point = [t4.iloc[0]['x'], t4.iloc[0]['y'], t4.iloc[0]['z']]
                            estimation_error = distance.euclidean(gt_point, s_point)
                            cont = False
                            write_result = False
                            k_cal_counter = -1
                            # There is a bug at opencv which outputs z < 0 values
                            # https://github.com/HKUST-Aerial-Robotics/VINS-Mono/issues/173
                            # ALSO
                            # https://stackoverflow.com/q/66268893/4955732
                            # Giray suspects this clause could be deleted.
                            if z_2 > 0.0:

                                if do_kalman_filter_predictions and i > config.kf_frame_required * 2:
                                    # forward pass of kalman prediction
                                    x, y, z, k_cal_counter = kalman_functions.kalman_predict(pre_id_1, fn, "recon2", False)
                                    if config.reverse_kf:
                                        # reverse pass of kalman prediction - requires a pre-existing 
                                        # result file
                                        x2_kf, y2_kf, z2_kf, k_cal_counter2_kf = kalman_functions.kalman_predict(pre_id_1, fn, "recon2", True)
                                        if k_cal_counter > k_cal_counter2_kf: # if available frame of forward run > reverse run
                                            pass
                                        else:
                                            x = x2_kf
                                            y = y2_kf
                                            z = z2_kf
                                            k_cal_counter = k_cal_counter2_kf
                                    # print("x and y: " + str(x) + ", " + str(y))
                                    # 2D distance:
                                    # dist_2_points = sqrt((x - gtx) ** 2 + (y - gty) ** 2)
                                    # 3D distance
                                    forecast_point = [x, y, z]
                                    # compare kalman prediction with 3d triangulation
                                    dist_2_points = distance.euclidean(s_point, forecast_point) 
                                    epipolar_distance_check = False
                                    if dist_2_points < kf_distance_threshold:
                                        write_result = True
                                        epipolar_distance_check = True
                                    else:
                                        if len(temp_c2) > 1:
                                            write_result = False
                                            cont = True
                                            ignore_candidate = True
                                            candidate_id_to_ignore = pre_id_1
                                        else:
                                            # write_result = False
                                            cont = False
                                            ignore_candidate = False
                                            if k_cal_counter <= config.kf_frame_required:
                                                # write_result = True
                                                pass
                                    # check that both one-way and two-way epipolar line
                                    # distances of the points are within a threshold. 
                                    # Here the threshold is hard-coded to 50 pixels.
                                    if config.kf_frame_required >= k_cal_counter and not epipolar_distance_check:
                                        if abs(pre_min_1) < 50 and abs(pre_min_2) < 50:
                                            write_result = True
                                            ignore_candidate = False
                                            cont = False
                                        else:
                                            if len(temp_c2) > 1:
                                                write_result = False
                                                ignore_candidate = True
                                                candidate_id_to_ignore = pre_id_1
                                                cont = True
                                if i <= config.kf_frame_required * 2:
                                    write_result = True
                                if not do_kalman_filter_predictions:
                                    write_result = True
                                if len(temp_c2) == 1:
                                    cont = False

                                if write_result and not ignore_candidate:
                                    df_recon = df_recon.append(
                                        pd.DataFrame(
                                            {"frame": pre_frame_1, "gtoid": t4.iloc[0]['oid'], "oid_1": pre_id_1,
                                             "oid_2": pre_id_2,
                                             "x": x_2, "y": y_2, "z": z_2, "gtx": gtx,
                                             "gty": gty, "gtz": gtz, "ee": estimation_error,
                                             "dt1": pre_min_1, "dt2": pre_min_2, "dt3": dist_2_points,
                                             "kcounter": k_cal_counter}, index=[0]))
                                    cont = False
                                    # THIS IS ONLY VALID FOR THE SYNTHETIC DATA AND STARLING DATA
                                    # The object IDs (pre_id_1 and pre_id_2) have been artificially set to be the same!!
                                    
                                    if t4.iloc[0]['oid'] == pre_id_1 and pre_id_1 == pre_id_2:
                                        scounter += 1
                                    else:
                                        fcounter += 1
                                    ccount += 1

                                if do_kalman_filter_predictions and not config.multi_run:
                                    df_recon.to_csv(f"result_files/{result_file_name}.csv", index=False)
                                # kalman filling Not implemented 
                                if call_kalman_fill:
                                    kalman_functions.kalman_fill("recon1", 10, i)
                                    call_kalman_fill = False
                                    path_to_check_1 = Path(f"result_files/{result_file_name}.csv")
                                    if path_to_check_1.exists():
                                        df_recon = pd.read_csv(path_to_check_1)

                                avg_ee += estimation_error
                                # print(res_tr)
                                if estimation_error > 1.0:
                                    pass
                        # if the two-way authentication does not giveany points
                        else:
                            # print("Unsuccessful match")
                            # Drop unsuccessful row

                            temp_c2 = temp_c2.drop(temp_c2.loc[(np.isclose(temp_c2['x'], cp1[0], 0.0001)) &
                                                               (np.isclose(temp_c2['y'], cp1[1], 0.0001))].index)
                            if len(temp_c2) <= 0:
                                cont = False
                            # print("At i: " + str(i) + " deleting point: " + str(cp1))
                            # print("New df: " + str(at_frame_c2))
                    else:
                        # if there is NO candidate found in the 20way auth epipolar line
                        cont = False
                else:
                    cont = False
    df_recon.to_csv(f"result_files/{result_file_name}.csv", index=False)
    # report all the performance diagnostics
    if (scounter + fcounter) > 0 and run_len != 0:
        print("Matching trajectory performance: " + str(scounter / (scounter + fcounter)) + " Scounter = " +
              str(scounter) + " Fcounter = " + str(fcounter))
        print("Total number of matched pairs: " + str(ccount) + ", rate: " + str((ccount / run_len) * 100.0))
        print("Average estimation error was:" + str(avg_ee / ccount))
        avg_ee = avg_ee / ccount
        matching_suc = scounter / (scounter + fcounter)

    return avg_ee, -1