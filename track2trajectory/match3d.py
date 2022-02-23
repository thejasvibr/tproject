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
import tqdm
import datetime as dt
from track2trajectory.prediction import kalman_predict
from track2trajectory.projection import triangulate_points_in3D

import pdb


def estimate_3d_points(camera1, camera2, all_2d_points: pd, all_3d_groundtruth, result_file_name,
                       fundamental_matrix,
                               **kwargs):
    '''
    Receives 2D projections from 2 cameras. Does epipolar matching, and after 10 
    frames applies Kalman Filtering, and gets robust pairing. 

    Does not assume that camera 1 and camera 2 have the same parameters. 

    Parameters
    ----------
    camera1 : camera.Camera
    camera2 : camera.Camera
    all_2d_points : pd.DataFrame
        2D projections of objects
    all_3d_groundtruth : pd.DataFrame
        Groundtruth 3D data. For comparison of the matched trajectories and 
        the groundtruth.
    result_file_name : str
        Final file name that the results will be saved in. 
    fundamental_matrix : np.array
        Fundamental matrix for camera 1 & 2
    kf_distance_threshold : float, optional
        Max allowable distance between Kalman Filter prediction. Defaults to
        0.3 m
    do_kalman_filter_predictions : bool, optional
        To run 3D Kalman predictions. Defaults to True
    kf_frame_required : int, optional 
        0.5 OF THE (CHECK!)Minimum number of frames required to run Kalman Filtering. 
        Defaults to 5/10 ---THIS NEEDS CHECKING!
    reverse_kf : bool, optional
        Defaults to False
    multi_run : bool, optional 
        Defaults to False

    Returns 
    -------
    avg_ee : float
        Average estimation error in metres
    -1 : int
        Significance of the value unclear (says Giray)

    Side effects
    ------------
    Write '{result_file_name}_{<timestamp>}.csv' with columns:
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
    * Split function into multiple parts -- it's too big
    * Rename the function
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
    
    See Also
    --------
    camera.Camera
    
    '''
    kf_distance_threshold = kwargs.get('kf_distance_threshold', 0.3) # metres
    do_kalman_filter_predictions = kwargs.get('do_kalman_filter_predictions', True)
    kf_frame_required = kwargs.get('kf_frame_required', 5)
    reverse_kf = kwargs.get('reverse_kf', False)
    multi_run = kwargs.get('multi_run', False)
    # save df with timestamp
    yyyymmdd = dt.datetime.now().strftime('%Y-%m-%d')


    all_2d_points = all_2d_points.sort_values('frame')
    cam1_2d_points = all_2d_points.loc[all_2d_points['cid'] == camera1.id] # trajectories cam 1
    cam2_2d_points = all_2d_points.loc[all_2d_points['cid'] == camera2.id] # trajectores cam 2

    run_len = len(cam1_2d_points)  # Number of frames and points

    # Benchmark values
    # see Sections 3.2-3.3 in Giray's thesis for more background
    reconstructed_3d = pd.DataFrame(columns=["frame", "gtoid", "oid_1", "oid_2",
                                     "x", "y", "z", "dt1", "dt2", "dt3"])
    scounter = 0 # successful pairings
    fcounter = 0 # failed pairings  
    ccount = 0 # Sum total of all points across all frames in Camera 1
    avg_ee = 0 

    for i in tqdm.trange(run_len):

        current_point_cam1 = np.float32([cam1_2d_points.iloc[i]['x'],
                                         cam1_2d_points.iloc[i]['y']])
        print(f'i:{i}, current_point_cam1:{current_point_cam1} \n')

        # Get proper frame values for comparison.
        frame_number = cam1_2d_points.iloc[i]['frame']
        at_frame_c1 = cam1_2d_points.loc[cam1_2d_points['frame'] == frame_number]
        at_frame_c2 = cam2_2d_points.loc[cam2_2d_points['frame'] == frame_number]

        call_kalman_fill = False

        if (scounter + fcounter) != 0:
            print("Matching perf: " + str(scounter / (scounter + fcounter)) + " Scounter = " +
                  str(scounter) + " Fcounter = " + str(fcounter) +
                  " Matching rate: " + str(((scounter + fcounter) / i) * 100.0))
            call_kalman_fill = False
        if at_frame_c2.empty and at_frame_c1.empty:
            continue

        cont = True # continue flag - becomes False after correspondence checks with all 
        # available pionts in frames are done

        temp_c2 = at_frame_c2.copy() # all points on camera 2 at this frame

        ignore_candidate = False
        candidate_id_to_ignore = 0
        while cont:

            if ignore_candidate:
                if len(temp_c2) > 1:
                    temp_c2 = temp_c2.drop(
                        temp_c2.loc[(np.isclose(temp_c2['oid'],
                                                candidate_id_to_ignore, 
                                                0.0001))].index)
                ignore_candidate = False
            dist_2_points = -1
            print(f'\n temp_c2: {temp_c2} \n')
            candidate_output1 = find_candidate(temp_c2, fundamental_matrix,
                                               camera2, camera1, current_point_cam1)
            #pdb.set_trace()
            cand_p_1, pre_min_1, pre_frame_1, pre_id_1, row_num1 = candidate_output1
            print(f'candidate_p1:{cand_p_1}')
            if not len(cand_p_1) != 0:
                continue 
            temp_c1 = at_frame_c1.copy() # all points on camera 1 at the current frame

            # the 'two-way authentication' is done here. The epipolar of the 'best point' in Cam 2 is 
            # is the projected onto Cam 1 - and the point closest to this epipolar is checked.
            candidate_output2 = find_candidate(temp_c1, fundamental_matrix, camera1,
                                                                   camera2, cand_p_1)
            cand_p_2, pre_min_2, pre_frame_2, pre_id_2, row_num1 = candidate_output2

            if not len(cand_p_2) != 0:
                cont = False
                continue
            # check that the '2-way auth' point on Cam1 is the same as the original 
            # candidate point
            print('delta: ',current_point_cam1[0]-cand_p_2[0],
                  current_point_cam1[1]-cand_p_2[1])
            if (np.isclose(current_point_cam1[0], cand_p_2[0], atol=0.0001) and
                    np.isclose(current_point_cam1[1], cand_p_2[1], atol=0.0001)):
                # print("Successful match")
                
                candp2 = cv2.undistortPoints(cand_p_2, camera1.i_mtrx,
                                             camera1.cof_mtrx,
                                             P=camera1.i_mtrx)
                candp1 = cv2.undistortPoints(cand_p_1, camera2.i_mtrx,
                                             camera2.cof_mtrx,
                                             P=camera2.i_mtrx)
                
                s_point = triangulate_points_in3D(candp2, candp1, 
                                                  camera1, camera2,
                                                             )
                # Get groun truth 3D value.
                oid_f = cam1_2d_points.iloc[i]['oid']
                t1 = cam1_2d_points.iloc[i]['frame']
                t2 = all_3d_groundtruth.loc[np.float32(all_3d_groundtruth['frame']) == np.float32(t1)]
                t4 = t2.loc[np.float32(t2['oid']) == oid_f]

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
                
                # Now TB suspects Giray was using the camera xyz
                # and so of course ran into issues when Z <= 0!
                # Needs to be replaced with y_2>0??
                x_2, y_2, z_2 = s_point
                print(f'\n s_point:{s_point}\n')
                if y_2 > 0.0:

                    if do_kalman_filter_predictions and i > kf_frame_required * 2:
                        # forward pass of kalman prediction
                        x, y, z, k_cal_counter = kalman_predict(pre_id_1, frame_number, "recon2", False)
                        if reverse_kf:
                            # reverse pass of kalman prediction - requires a pre-existing 
                            # result file
                            x2_kf, y2_kf, z2_kf, k_cal_counter2_kf = kalman_predict(pre_id_1, frame_number, "recon2", True)
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
                                if k_cal_counter <= kf_frame_required:
                                    # write_result = True
                                    pass
                        # check that both one-way and two-way epipolar line
                        # distances of the points are within a threshold. 
                        # Here the threshold is hard-coded to 50 pixels.
                        if kf_frame_required >= k_cal_counter and not epipolar_distance_check:
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
                    if i <= kf_frame_required * 2:
                        write_result = True
                    if not do_kalman_filter_predictions:
                        write_result = True
                    if len(temp_c2) == 1:
                        cont = False

                    if write_result and not ignore_candidate:
                        reconstructed_3d = reconstructed_3d.append(
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

                    if do_kalman_filter_predictions and not multi_run:
                        reconstructed_3d.to_csv(f'{result_file_name}_{yyyymmdd}.csv', index=False)

                    avg_ee += estimation_error
                    # print(res_tr)
                    if estimation_error > 1.0:
                        pass
            # if the two-way authentication does not giveany points
            else:
                # print("Unsuccessful match")
                # Drop unsuccessful row

                temp_c2 = temp_c2.drop(temp_c2.loc[(np.isclose(temp_c2['x'], cand_p_1[0], 0.0001)) &
                                                   (np.isclose(temp_c2['y'], cand_p_1[1], 0.0001))].index)
                if len(temp_c2) <= 0:
                    cont = False
                # print("At i: " + str(i) + " deleting point: " + str(cand_p_1))
                # print("New df: " + str(at_frame_c2))
        else:
            # if there is NO candidate found in the 2-way auth epipolar line
            cont = False
            print('meep')

    reconstructed_3d.to_csv(f"{result_file_name}_{yyyymmdd}.csv", index=False)
    # report all the performance diagnostics
    if (scounter + fcounter) > 0 and run_len != 0:
        print("Matching trajectory performance: " + str(scounter / (scounter + fcounter)) + " Scounter = " +
              str(scounter) + " Fcounter = " + str(fcounter))
        print("Total number of matched pairs: " + str(ccount) + ", rate: " + str((ccount / run_len) * 100.0))
        print("Average estimation error was:" + str(avg_ee / ccount))
        avg_ee = avg_ee / ccount
        matching_suc = scounter / (scounter + fcounter)

    return avg_ee, -1

if __name__ == '__main__':
    # test the fundamental camera relation
   
    # output = estimate_3d_points(cam1, cam2, 
    #                          all_2dpoints, points_in_3d,
    #                          'output_testresuts', fundamatrix_cam2fromcam1,
    #                          do_kalman_filter_predictions=True)

                                                                     
                                                                     
                                                                     
                                                                     