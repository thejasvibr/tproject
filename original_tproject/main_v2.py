'''
# References & used resources for this implementation:
# https://docs.opencv.org/3.4/
# https://stackoverflow.com/
# https://www.geeksforgeeks.org/
# Hartley, Richard, and Andrew Zisserman. Multiple View Geometry in Computer Vision., 2004.
# Mixed-reality spatial configuration with a zed mini stereoscopic camera, 2018, Chotrov, D et al.
# Determining the epipolar geometry and its uncertainty: A review, 1998, Zhang et al.




Original code written by Giray Tandogan 


'''
import math
import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

from cv2 import CV_32FC1
from scipy.spatial import distance
from pathlib import Path
import cv2
from xml.dom import minidom
import os
import glob
from math import cos, sin, sqrt, pi
import sys
import config
import classes
import kalman_functions
import data_functions
import visualizations

PY3 = sys.version_info[0] == 3
if PY3:
    long = int

pd.options.display.float_format = '{:.2f}'.format

# Config setup
error_code = config.error_code
visualization = config.visualizations
how_many_objects_to_create = config.how_many_objects_to_create
force_calc = config.force_calc
create_objects = config.create_objects
do_estimations = config.do_estimations
do_kalman_filter_predictions = config.do_kalman_filter_predictions
gt_construction = config.gt_construction


def calc_distance_point_line(p, q, r):
    '''
    p = First point of the line, 3d.
    q = Second point of the line, 3d.
    r = A point, 3d.
    Example:
    p = np.array([0, 0, -10])
    q = np.array([25, 0, 0])
    r = np.array([10, 10, 10])
    '''

    def t(p, q, r):
        x = p - q
        return np.dot(r - q, x) / np.dot(x, x)

    def d(p, q, r):
        return np.linalg.norm(t(p, q, r) * (p - q) + q - r)

    return d(p, q, r)



def calcFundamentalMatrix(camera_1_o, camera_2_o):
    '''
    Calculates the fundamental matrix 

    Parameters
    ----------
    camera_1_o, camera_2_o : classes.Camera instances 
    
    Returns
    -------
    second : np.array 
        2D array with the fundamental matrix

    Notes
    -----
    For details please check page 246 "Multiple View Geometry".
    Used formula F = [e']_x * P' * p^+
                      
    The fundamental matrix is a 3X3 matrix which maps a point on cam1 onto potential 
    points on camera 2. The epipolar line can be computed from the fundamental matrix

    See Also 
    --------
    find_candidate
    cv2.computeCorrespondEpilines
    '''
    rt = camera_1_o.r_mtrx.transpose()
    rt = np.negative(rt)
    cpos = np.matmul(rt, camera_1_o.t_mtrx)
    relpos = [cpos[0], cpos[1], cpos[2], 1.0]
    e_p = np.matmul(camera_2_o.cm_mtrx, relpos)
    e_p_cross = [[0.0, e_p[2] * -1.0, e_p[1]],
                 [e_p[2], 0.0, e_p[0] * -1.0],
                 [e_p[1] * -1.0, e_p[0], 0.0]]
    first = np.matmul(e_p_cross, camera_2_o.cm_mtrx)
    p_plus = camera_1_o.cm_mtrx.transpose()
    p_plus_2 = np.linalg.inv(np.matmul(camera_1_o.cm_mtrx, p_plus))
    p_plus_3 = np.matmul(p_plus, p_plus_2)
    second = np.matmul(first, p_plus_3)
    return second


def project_to_2d_and_3d_no_id(gt_3d_df, camera_1: classes.Camera, camera_2: classes.Camera):
    proj_2d_df = pd.DataFrame(columns=["frame", "oid", "cid", "x", "y"])
    path_to_check_1 = Path("gt_files/proj_2d.csv")
    if path_to_check_1.exists() and not force_calc:
        proj_2d_df = pd.read_csv(path_to_check_1)
    else:
        camera_obj = camera_1
        camera_obj_2 = camera_2
        change_object_id_1 = False
        change_object_id_2 = False
        fail_counter = 0
        oid_c_1 = 0
        oid_c_2 = 1

        for i in range(len(gt_3d_df)):
            # Set values.
            x_temp = gt_3d_df.iloc[i]['x']
            y_temp = gt_3d_df.iloc[i]['y']
            z_temp = gt_3d_df.iloc[i]['z']
            frame_temp = gt_3d_df.iloc[i]['frame']
            obj_xyz_temp = [x_temp, y_temp, z_temp]
            obj_xyz_temp = numpy.float32(obj_xyz_temp)

            # Project Points from 3D to 2D
            r_m, _ = cv2.Rodrigues(camera_obj.r_mtrx)
            oi, _ = cv2.projectPoints(obj_xyz_temp, r_m,
                                      camera_obj.t_mtrx, camera_obj.i_mtrx, camera_obj.cof_mtrx)
            r_m_2, _ = cv2.Rodrigues(camera_obj_2.r_mtrx)
            oi_2, _ = cv2.projectPoints(obj_xyz_temp, r_m_2,
                                        camera_obj_2.t_mtrx, camera_obj_2.i_mtrx, camera_obj_2.cof_mtrx)

            # Check if object is in image plane
            if 1920.0 >= oi[0][0][0] >= 0.0 and 0.0 <= oi[0][0][1] <= 1080.0:

                # Undistort 2D points.
                # proj_2d = cv2.undistortPoints(oi, camera_obj.i_mtrx, camera_obj.cof_mtrx, P=camera_obj.i_mtrx)
                # Set 2D points to data frame.
                proj_2d_df = proj_2d_df.append(pd.DataFrame({"frame": [frame_temp], "oid": [oid_c_1],
                                                             "x": [oi[0][0][0]], "y": [oi[0][0][1]],
                                                             "cid": 0}))
                change_object_id_1 = True
            elif change_object_id_1:
                change_object_id_1 = False

                oid_c_1 += 1
                if oid_c_1 == oid_c_2:
                    oid_c_1 += 1

                print("New object c1: " + str(oid_c_1))
                fail_counter += 1
            else:
                fail_counter += 1

            if 1920.0 >= oi_2[0][0][0] >= 0.0 and 0.0 <= oi_2[0][0][1] <= 1080.0:
                # Set 2D points to data frame.
                proj_2d_df = proj_2d_df.append(pd.DataFrame({"frame": [frame_temp], "oid": [oid_c_2],
                                                             "x": [oi_2[0][0][0]], "y": [oi_2[0][0][1]],
                                                             "cid": 1}))
                change_object_id_2 = True
            elif change_object_id_2:
                change_object_id_2 = False

                oid_c_2 += 1
                if oid_c_1 == oid_c_2:
                    oid_c_2 += 1
                print("New object c2: " + str(oid_c_1))
                fail_counter += 1
            else:
                fail_counter += 1

        # del x_temp, y_temp, z_temp, frame_temp, oid, obj_xyz_temp, oi, \
        #    proj_2d, projected_3d_x, projected_3d_y, projected_3d_z

        # Write dataframe to csv file.
        if path_to_check_1.exists() and camera_obj.id != 0:
            proj_2d_df.to_csv(path_to_check_1, mode='a', header=False, index=False)
        else:
            proj_2d_df.to_csv(path_to_check_1, index=False)
    print("Failed total points: " + str(fail_counter))
    return proj_2d_df


def estimate_3d_points(camera1, camera2, df_2d: pd, df_3d_gt, result_file_name, fm_1):
    '''
    Receives 2D projections from 2 cameras. Does epipolar matching, and after 10 
    frames applies Kalman Filtering, and gets robust pairing. 

    Does not assume that camera 1 and camera 2 have the same parameters. 

    Parameters
    ----------
    camera1 : classes.Camera
    camera2 : classes.Camera
    df_2d : pd.DataFrame
        2D projections of objects
    df_3d_gt : pd.DataFrame
        Groundtruth 3D data. For comparison of the matched trajectories and 
        the groundtruth.
    result_file_name : str
        Final file name that the results will be saved in. 
    fm_1 : np.array
        Fundamental matrix for camera 1

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

    for i in range(0, run_len):
        temp_p = np.float32([df_t.iloc[i]['x'], df_t.iloc[i]['y']])
        # temp_p = df_t.iloc[i:i+1]
        # Get proper frame values for comparison.
        fn = df_t.iloc[i]['frame']
        at_frame_c2 = df_t_2.loc[df_t_2['frame'] == fn]

        at_frame_c1 = df_t.loc[df_t['frame'] == fn]
        check_if = int(i % (run_len / 100.0 * 1.0)) # progress bar
        call_kalman_fill = False
        if check_if == 0:
            print("Estimating progress: " + str(int(i / run_len * 100)) + "%")

            if (scounter + fcounter) != 0:
                print("Matching perf: " + str(scounter / (scounter + fcounter)) + " Scounter = " +
                      str(scounter) + " Fcounter = " + str(fcounter) +
                      " Matching rate: " + str(((scounter + fcounter) / i) * 100.0))
                call_kalman_fill = False
        if not at_frame_c2.empty and not at_frame_c1.empty:

            cont = True # continue flag - becomes False after correspondence checks with all 
            # available pionts in frames are done
            temp_c2 = at_frame_c2.copy() # all points on camera 2 at this frame

            # p1 = numpy.float64([cand1[0][0][0], cand1[0][0][1], 1])
            # p2 = numpy.float64([cand2[0][0][0], cand2[0][0][1], 1])
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
                                    if dist_2_points < config.kf_distance_threshold:
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


def reconstruct_trajectories(camera1, camera2, df_2d: pd, df_3d_gt, result_file_name):
    '''
    Parameters
    ----------
    camera1, camera2: classes.Camera instances 
    df_2d : pd.DataFrame
        Matched points with object ids across the 2 cameras
    df_3d_gt: pd.DataFrame 
        Groundtruth 3D data
    result_file_name : string
        Target output file name
    '''
    df_2d = df_2d.sort_values('frame')
    df_t = df_2d.loc[df_2d['cid'] == camera1.id]
    df_t_2 = df_2d.loc[df_2d['cid'] == camera2.id]

    run_len = len(df_t)  # Number of frames.

    # Benchmark values
    df_recon = pd.DataFrame(columns=["frame", "gtoid", "oid_1", "oid_2", "x", "y", "z", "dt1", "dt2", "dt3"])
    scounter = 0
    fcounter = 0
    ccount = 0
    avg_ee = 0
    find_trajectory(0, [0, 0, 0], True, 0)
    df1numpya = numpy.float64(df_t[['x', 'y']].to_numpy())
    df2numpya = numpy.float64(df_t_2[['x', 'y']].to_numpy())
    cand1 = cv2.undistortPoints(df1numpya, camera1.i_mtrx, camera1.cof_mtrx,
                                P=camera1.i_mtrx)
    cand2 = cv2.undistortPoints(df2numpya, camera2.i_mtrx, camera2.cof_mtrx,
                                P=camera2.i_mtrx)

    if len(cand1) != len(cand2):
        if len(cand1) > len(cand2):
            cand1 = cand1[:len(cand2)]
        else:
            cand2 = cand2[:len(cand1)]

    outfm = cv2.findFundamentalMat(cand1, cand2,
                                   method=cv2.RANSAC, ransacReprojThreshold=3, confidence=0.99, maxIters=10000)

    for i in range(0, run_len):
        temp_p = np.float32([df_t.iloc[i]['x'], df_t.iloc[i]['y']])
        # temp_p = df_t.iloc[i:i+1]
        # Get proper frame values for comparison.
        fn = df_t.iloc[i]['frame']
        at_frame_c2 = df_t_2.loc[df_t_2['frame'] == fn]

        at_frame_c1 = df_t.loc[df_t['frame'] == fn]
        check_if = int(i % (run_len / 100.0 * 1.0))
        call_kalman_fill = False # not used anywhere - flag to activate  Kalman based
        # 'filling' or interpolation of 2D points that were not matched to a trajectory
        if check_if == 0:
            print("Reconstruction progress: " + str(int(i / run_len * 100)) + "%")

            if (scounter + fcounter) != 0:
                print("Matching perf: " + str(scounter / (scounter + fcounter)) + " Scounter = " +
                      str(scounter) + " Fcounter = " + str(fcounter) +
                      " Matching rate: " + str(((scounter + fcounter) / i) * 100.0))
                call_kalman_fill = False
        if not at_frame_c2.empty and not at_frame_c1.empty:

            cont = True
            temp_c2 = at_frame_c2.copy()

            while cont:
                current_id = df_t.iloc[i]['oid']
                at_frame_c2 = at_frame_c2.loc[at_frame_c2['oid'] == current_id]
                if not at_frame_c2.empty:
                    temp_p2 = np.float32([at_frame_c2.iloc[0]['x'], at_frame_c2.iloc[0]['y']])

                    candp2 = cv2.undistortPoints(temp_p, camera1.i_mtrx, camera1.cof_mtrx,
                                                 P=camera1.i_mtrx)
                    candp1 = cv2.undistortPoints(temp_p2, camera2.i_mtrx, camera2.cof_mtrx,
                                                 P=camera2.i_mtrx)
                    tri_res = cv2.triangulatePoints(camera1.cm_mtrx, camera2.cm_mtrx,
                                                    candp2, candp1)
                    x_2 = tri_res[0] / tri_res[3]
                    y_2 = tri_res[1] / tri_res[3]
                    z_2 = tri_res[2] / tri_res[3]
                    # Get benchmark rotate3D value.
                    oid_f = df_t.iloc[i]['oid']
                    t1 = df_t.iloc[i]['frame']
                    t2 = df_3d_gt.loc[np.float32(df_3d_gt['frame']) == np.float32(t1)]
                    t4 = t2.loc[np.float32(t2['oid']) == oid_f]

                    xf = x_2
                    yf = y_2
                    zf = z_2
                    s_point = [x_2, y_2, z_2]

                    gtx = t4.iloc[0]['x']
                    gty = t4.iloc[0]['y']
                    gtz = t4.iloc[0]['z']

                    gt_point = [t4.iloc[0]['x'], t4.iloc[0]['y'], t4.iloc[0]['z']]
                    estimation_error = distance.euclidean(gt_point, s_point)

                    df_recon = df_recon.append(
                        pd.DataFrame(
                            {"frame": fn, "gtoid": t4.iloc[0]['oid'], "oid_1": current_id,
                             "oid_2": current_id,
                             "x": x_2, "y": y_2, "z": z_2, "gtx": gtx,
                             "gty": gty, "gtz": gtz, "ee": estimation_error,
                             "dt1": 0, "dt2": 0, "dt3": 0,
                             "kcounter": 0}, index=[0]))
                cont = False
    df_recon.to_csv(f"result_files/{result_file_name}.csv", index=False)


def match_2d_trajectories(camera1: classes.Camera, camera2: classes.Camera, file_name, file_name_2, results_file_name,
                          gt_3d_df):
    path_to_check_1 = Path(f"result_files/{file_name}.csv")
    path_to_check_2 = Path(f"result_files/{file_name_2}.csv")
    path_to_check_3 = Path(f"gt_files/proj_2d.csv")

    if path_to_check_1.exists() and path_to_check_2.exists() and path_to_check_3:
        df_2d = pd.read_csv(path_to_check_1)
        df_2d_gt = pd.read_csv(path_to_check_3)
        df_2d = df_2d.sort_values('frame')
        df_2d_gt = df_2d_gt.sort_values('frame')
        df_2d_gt_c1 = df_2d_gt.loc[df_2d_gt['cid'] == camera1.id]
        df_2d_gt_c2 = df_2d_gt.loc[df_2d_gt['cid'] == camera2.id]
        # df_2d_c1 = df_2d.loc[df_2d['cid'] == camera1.id]
        # df_2d_c2 = df_2d.loc[df_2d['cid'] == camera2.id]
        df_of_tid = df_2d.oid_1.unique()
        avg_ee = 0
        m_success = 0
        id_perf = 0

        for i in df_of_tid:
            print("Matching trajectories i at: " + str(i))
            df_recon = pd.DataFrame(columns=["frame", "oid", "cid", "x", "y"])
            df_2d_c1 = df_2d.loc[df_2d['oid_1'] == i]
            vcounter = df_2d_c1['oid_2'].value_counts()
            most_common_id = vcounter.idxmax()
            sum_of_ids = vcounter.sum()
            max_id = vcounter.max()
            # print("Successful matching of correspoind points: " + str(max_id / sum_of_ids))
            id_perf += max_id / sum_of_ids
            rows1 = df_2d_gt_c1.loc[df_2d_gt_c1['oid'] == i]
            rows2 = df_2d_gt_c2.loc[df_2d_gt_c2['oid'] == most_common_id]

            df_recon = df_recon.append(rows1)
            df_recon = df_recon.append(rows2)
            name = results_file_name + str(i)
            t_avg_ee, t_m_success = estimate_3d_points(camera_1, camera_2, df_recon, gt_3d_df, name)
            avg_ee += t_avg_ee
            m_success += t_m_success
            pass
        avg_ee = avg_ee / len(df_of_tid)
        m_success = m_success / len(df_of_tid)
        id_perf = id_perf / len(df_of_tid)
        # print("End of 2D t matching. Average ee was: " + str(avg_ee).format() + " Average matching: " + str(m_success) +
        #      " ID matching process before trajectory matching was: " + str(id_perf))
        pass
    else:
        return -1


def find_candidate(df_2d_c1, fund_matrix, camera1: classes.Camera, camera2: classes.Camera, candidate_point,
                   known_depth_for_camera, threshold, rec):
    '''
    Gives the 2D point closest to the epipolar line projected from the reference camera.

    Parameters
    ----------
    df_2d_c1 : pd.DataFrame
        xy coordinates for on the camera 'reference' camera  (assigned to number 1)
    fund_matrix: np.array
        Fundamental matrix. 
    camera1, camera2 : classes.Camera instances
    candidate_point :  np.array (np.float32)
    known_depth_for_camera : float >0 
        Not used any more in the latest implementation -- but 'works'
    threshold : float 
        Not used any more in the latest implementation -- but 'works'
    rec : bool
        True/False - not used any more -- but 'works'
        Should ALWAYS BE FALSE!!!!!!!!!!!!!!!!!!!!!!!
    
    Returns 
    -------
    pre_p : np.array
        x,y coordinates of the point with min distance to epipolar line on the 
        other camera
    pre_min : float
        The perpendicular Euclidean distance of the best match to the epipolar line
    pre_frame : int 
        Frame number. This is here more for diagnostic reasons as the pre_frame should 
        match the same frame number as df_2d_c1
    pre_id : int
        Object id of the best point in the 'other' camera 
    row_num : int 
        The row number of the best matchign point in df_2d_c1. This too is here more 
        for diagnostic reasons.
        
    See Also
    --------
    calcFundamentalMatrix
    
    
    '''
    run_len = len(df_2d_c1)
    pre_min = math.inf
    x_cand, y_cand, pre_frame, pre_id = -1, -1, -1, -1
    pre_p = []
    row_num = -1

    for k in range(run_len):
        temp_p = np.float32([df_2d_c1.iloc[k]['x'], df_2d_c1.iloc[k]['y']])
        if len(candidate_point) != 0:
            points_undistorted_2 = cv2.undistortPoints(candidate_point, camera2.i_mtrx, camera2.cof_mtrx,
                                                       P=camera2.i_mtrx)
        else:
            return [], -1, -1, -1, -1


        # Set camera ids for computeCorrespondEpilines()
        if camera1.id < camera2.id:
            cid = 1
        else:
            cid = 2

        # Find epipolar line values: a,b,c. From ax+by+c=0.
        df1numpya = numpy.float64(df_2d_c1[['x', 'y']].to_numpy())
        epilines = cv2.computeCorrespondEpilines(df1numpya, cid, fund_matrix)
        v_a = epilines[k][0][0]
        v_b = epilines[k][0][1]
        v_c = epilines[k][0][2]
        dist_fund_point = (points_undistorted_2[0][0][0] * v_a) + (points_undistorted_2[0][0][1] * v_b) + v_c

        # if dist_2d_2 < pre_min and dist_2d_2 < threshold:
        if abs(dist_fund_point) < pre_min:
            pre_min = abs(dist_fund_point)
            pre_id = df_2d_c1.iloc[k]['oid']
            pre_frame = df_2d_c1.iloc[k]['frame']
            pre_p = temp_p
            row_num = k
        else:
            pass

    if pre_min > 0.1 and known_depth_for_camera >= 1 and rec:
        pre_p2, pre_min2, pre_frame2, pre_id2, row_num2 = find_candidate(df_2d_c1, fund_matrix, camera1, camera2,
                                                                         candidate_point, known_depth_for_camera - 1,
                                                                         threshold, rec)
        if pre_min2 < pre_min:
            pre_p, pre_min, pre_frame, pre_id, row_num = pre_p2, pre_min2, pre_frame2, pre_id2, row_num2

    return pre_p, pre_min, pre_frame, pre_id, row_num


def find_trajectory(frame_number, current_candidate_point, create_new_file, dist_threshold):
    trj_df = pd.DataFrame(columns=["frame", "tid", "x", "y", "z", "bfc", "matched"])
    path_to_check_1 = Path("result_files/trajectories.csv")

    if create_new_file:
        trj_df.to_csv(path_to_check_1, index=False)
        return 1

    if path_to_check_1.exists() and not create_new_file and frame_number != 0:
        trj_df = pd.read_csv(path_to_check_1)
    elif path_to_check_1.exists() and frame_number == 0:
        trj_df = pd.read_csv(path_to_check_1)
        leng1 = len(trj_df)
        if leng1 == 0:
            trj_df = trj_df.append(pd.DataFrame({"frame": [frame_number], "tid": [0],
                                                 "x": current_candidate_point[0],
                                                 "y": current_candidate_point[1],
                                                 "z": current_candidate_point[2], "bfc": [0], "matched": [0]}))
            trj_df.to_csv(path_to_check_1, index=False)
            return 1
        else:
            trj_df.to_csv(path_to_check_1, index=False)
            return 1
    else:
        trj_df.to_csv(path_to_check_1, index=False)
        return 0
    pre_dist = 99999
    found_flag = True
    cpx, cpy, cpz = -1, -1, -1
    found_index = -1
    candidate_positions = trj_df.loc[trj_df['frame'] == (frame_number - 1)]
    candidate_positions = candidate_positions.loc[trj_df['matched'] == -1]
    lcand = len(candidate_positions)
    df_of_tid = trj_df.tid.unique()
    back_frame_counter = 0
    back_frame_limit = 1
    len_tr = len(trj_df)
    adaptive_threshold = 0.1

    while len_tr > 0 and found_flag and back_frame_counter < back_frame_limit:
        while found_flag and lcand > 0:
            for i in range(lcand):
                cpx = np.float32(candidate_positions.iloc[i]["x"])
                cpy = np.float32(candidate_positions.iloc[i]["y"])
                cpz = np.float32(candidate_positions.iloc[i]["z"])
                cp = [cpx, cpy, cpz]
                dist = distance.euclidean(cp, current_candidate_point)

                if dist < dist_threshold and dist < pre_dist:
                    pre_dist = dist
                    found_flag = False
                    found_index = i
            if found_flag:
                dist_threshold += 0.1
            if found_flag and dist_threshold > adaptive_threshold and back_frame_counter < back_frame_limit:
                back_frame_counter += 1
                candidate_positions = trj_df.loc[trj_df['frame'] == (frame_number - 1 - back_frame_counter)]
                lcand = len(candidate_positions)
            if found_flag and dist_threshold > adaptive_threshold and back_frame_counter > back_frame_limit:
                pass

        if len(candidate_positions) > 0:
            trj_df_f = trj_df.loc[(np.isclose(trj_df['x'], candidate_positions.iloc[found_index]['x'], 0.0001)) &
                                  (np.isclose(trj_df['y'], candidate_positions.iloc[found_index]['y'], 0.0001)) &
                                  (np.isclose(trj_df['z'], candidate_positions.iloc[found_index]['z'], 0.0001))]
            if len(trj_df_f) > 0:
                tid_match = trj_df_f.iloc[0]['tid']
                trj_df = trj_df.append(pd.DataFrame({"frame": [frame_number], "tid": [tid_match],
                                                     "x": current_candidate_point[0],
                                                     "y": current_candidate_point[1],
                                                     "z": current_candidate_point[2], "bfc": [back_frame_counter],
                                                     "matched": [0]}))
                trj_df_f = trj_df.loc[(np.isclose(trj_df['x'], cpx, 0.0001)) &
                                      (np.isclose(trj_df['y'], cpy, 0.0001)) &
                                      (np.isclose(trj_df['z'], cpz, 0.0001))].index
                trj_df.loc[trj_df_f, 'matched'] = 1
                trj_df.to_csv(path_to_check_1, index=False)
                return 1

        back_frame_counter += 1
        candidate_positions = trj_df.loc[trj_df['frame'] == (frame_number - 1 - back_frame_counter)]
        lcand = len(candidate_positions)
        adaptive_threshold += 0.5

    if len(df_of_tid) > 0:
        tid_match = int(df_of_tid[len(df_of_tid) - 1]) + 1
    else:
        tid_match = 1
    trj_df = trj_df.append(pd.DataFrame({"frame": [frame_number], "tid": [tid_match],
                                         "x": current_candidate_point[0],
                                         "y": current_candidate_point[1],
                                         "z": current_candidate_point[2], "bfc": [back_frame_counter], "matched": [0]}))
    trj_df.to_csv(path_to_check_1, index=False)


def find_trajectory_2d(df_2d, which_camera: classes.Camera, next_frames_limit, dist_threshold, file_name, start_frame,
                       deep_check, f_check):
    n_df_2d = df_2d.sort_values('frame')
    n_df_2d = n_df_2d.loc[n_df_2d['cid'] == which_camera.id]
    trj_2d_df = pd.DataFrame(columns=["frame", "tid", "x", "y", "matched"])
    path_to_check_1 = Path(f"result_files/{file_name}.csv")
    trj_2d_df.to_csv(path_to_check_1, index=False)
    tmp_2d = n_df_2d.loc[n_df_2d['frame'] == start_frame]

    for i in range(len(tmp_2d)):
        current_point = tmp_2d.iloc[i]
        cp = np.array([current_point['x'], current_point['y']])
        cur_frame_num = int(tmp_2d.iloc[i]['frame'])
        trj_2d_df = pd.read_csv(path_to_check_1)
        l_csv_file = len(trj_2d_df)
        if cur_frame_num == start_frame and l_csv_file <= 0:
            trj_2d_df = trj_2d_df.append(pd.DataFrame({"frame": [cur_frame_num], "tid": [0],
                                                       "x": cp[0], "y": cp[1], "matched": [-1]}))
            trj_2d_df.to_csv(path_to_check_1, index=False)
            pass

        elif cur_frame_num == start_frame and l_csv_file >= 1:
            pre_tid = trj_2d_df.iloc[l_csv_file - 1]['tid'] + 1
            trj_2d_df = trj_2d_df.append(pd.DataFrame({"frame": [cur_frame_num], "tid": [pre_tid],
                                                       "x": cp[0], "y": cp[1], "matched": [-1]}))
            trj_2d_df.to_csv(path_to_check_1, index=False)
            pass
    tmp_2d_c1 = n_df_2d.loc[n_df_2d['cid'] == which_camera.id]
    tmp_2d_c1['matched'] = -1
    num_of_frames = tmp_2d_c1.frame.unique()
    len_2d_c1 = len(num_of_frames)
    lowest_dist = math.inf
    for k in range(len_2d_c1):
        trj_2d_df_o = pd.read_csv(path_to_check_1)
        trj_2d_df = trj_2d_df_o.loc[(trj_2d_df_o['frame'] == k - 1)]
        l_csv_file = len(trj_2d_df)
        for t in range(l_csv_file):

            cur_frame_num = trj_2d_df.iloc[t]['frame']
            if cur_frame_num == 53 and deep_check:
                print("")
            cur_tid_num = trj_2d_df.iloc[t]['tid']
            next_2d_data = tmp_2d_c1.loc[(tmp_2d_c1['frame'] == k) & (tmp_2d_c1['matched'] == -1)]
            len_next_2d_d = len(next_2d_data)
            if len_next_2d_d < l_csv_file:
                trj_avoid_id_f = tmp_2d_c1.loc[(tmp_2d_c1['frame'] == k) & (tmp_2d_c1['matched'] != -1)]
                if len(trj_avoid_id_f) != 0:
                    for av in range(len(trj_avoid_id_f)):
                        avoid_id = trj_avoid_id_f.iloc[av]['matched']
                        if cur_tid_num == avoid_id:
                            len_next_2d_d = -1
            cp = np.array([trj_2d_df.iloc[t]['x'], trj_2d_df.iloc[t]['y']])
            pre_min = math.inf
            pre_i = -1
            pre_c = np.array([-1, -1])
            pre_pc = np.array([-1, -1])
            dist2 = math.inf
            dist3 = 9999
            dist4 = 1111
            rdist = math.inf
            dist_dif_lim = math.inf

            for z in range(len_next_2d_d):
                over_l = False
                ncp = np.array([next_2d_data.iloc[z]['x'], next_2d_data.iloc[z]['y']])
                dist = distance.euclidean(cp, ncp)
                dist_dif_lim = abs(dist - pre_min)
                if dist < pre_min:
                    # dist2 = abs(ncp - pre_c)
                    dist2 = distance.euclidean(ncp, pre_c)

                    if pre_i != -1 and deep_check:
                        dist3 = distance.euclidean(cp, ncp)
                        dist4 = distance.euclidean(cp, pre_c)
                        #  print(abs(dist3 - dist4))
                    # TODO: fix with proper value assignments.
                    if over_l:
                        pre_min = dist
                        pre_pc = cp
                        pre_i = z
                    else:
                        pre_pc = pre_c
                        pre_c = ncp
                        pre_min = dist
                        pre_i = z
            if pre_i != -1:
                rdist = abs(dist3 - dist4)
                if rdist < lowest_dist:
                    lowest_dist = rdist
            if (dist2 < 20 or rdist < 20 or dist_dif_lim < 60) and k > 10:
                if deep_check:

                    print("Possible overlap. Frame: " + str(k))
                    if dist2 < 100:
                        print("Candidate is too close to others.")
                    if rdist < 20:
                        print("Candidates' distances are too close.")
                    if dist_dif_lim < 50:
                        print("Candidates' distances are very similar.")
                    if which_camera.id == 0:
                        s_camera = camera_2
                    elif which_camera.id == 1:
                        s_camera = camera_1
                    else:
                        pass

                    if dist2 < lowest_dist:
                        # lowest_dist = dist2
                        pass

                    find_trajectory_2d(df_2d, s_camera, 2, 5, "trj_other_camera", k - 10, False, False)
                    s_df = df_2d.loc[(df_2d['frame'] == cur_frame_num) & (df_2d['cid'] == s_camera.id)]
                    if f_check:
                        cand_p, pre_min, _, _, _ = find_candidate(s_df, s_camera, which_camera, ncp, 3, 50, False)
                        cand_p2, pre_min2, _, _, _ = find_candidate(s_df, s_camera, which_camera, pre_c, 3, 50, False)
                        print("Candidate points: " + str(cand_p2) + ", " + str(cand_p) + "; was comparing:" + str(
                            pre_c) + ", " + str(ncp))
                        # cand_p2, pre_min2, _, _, _ = find_candidate(next_2d_data, which_camera, s_camera,
                        # cand_p, 3, 50) print("Candidate point: " + str(cand_p2) + ", was comparing:" + str(
                        # pre_c) + ", " + str(ncp))
                        cand_p3, pre_min, _, _, _ = find_candidate(next_2d_data, which_camera, s_camera, cand_p, 3,
                                                                   50, False)
                        cand_p4, pre_min2, _, _, _ = find_candidate(next_2d_data, which_camera, s_camera, cand_p2,
                                                                    3, 50, False)
                        print("Candidate points: " + str(cand_p3) + ", " + str(cand_p4) + "; was comparing:" + str(
                            pre_c) + ", " + str(ncp))
                        if np.isclose(cand_p3[0], cand_p4[0], atol=0.0001) and np.isclose(cand_p4[1], cand_p4[1],
                                                                                          atol=0.0001) and f_check:
                            pre_c = cand_p4

                    path_to_check_trj = Path(f"result_files/trj_other_camera.csv")
                    if path_to_check_trj.exists():
                        r_trj = pd.read_csv(path_to_check_trj)
                        list_m_ids = []
                        # Go back x frames to find proper match for trajectory id.
                        points_dif = 0
                        for tb in range(5):
                            s_df_2 = df_2d.loc[
                                (df_2d['frame'] == cur_frame_num - 1 - tb) & (df_2d['cid'] == s_camera.id)]
                            s_df_3 = df_2d.loc[
                                (df_2d['frame'] == cur_frame_num - 1 - tb) & (df_2d['cid'] == which_camera.id)]
                            trj_2d_df_prev = pd.read_csv(path_to_check_1)
                            trj_2d_df_prev = trj_2d_df_prev.loc[(trj_2d_df_prev['frame'] == cur_frame_num - 1 - tb) &
                                                                (trj_2d_df_prev['tid'] == cur_tid_num)]
                            pre_point = np.array([trj_2d_df_prev.iloc[0]['x'], trj_2d_df_prev.iloc[0]['y']])
                            cand_p5, pre_min, _, _, _ = find_candidate(s_df_2, s_camera, which_camera,
                                                                       pre_point, 10, 50, True)
                            cand_p6, pre_min2, _, _, _ = find_candidate(s_df_3, which_camera, s_camera, cand_p5,
                                                                        10, 50, True)
                            if len(cand_p5) != 0 and len(cand_p6) != 0:
                                points_dif += distance.euclidean(pre_point, cand_p6)
                                trj_matched_data = r_trj.loc[(r_trj['frame'] == cur_frame_num - 1 - tb) &
                                                             (np.isclose(r_trj['x'], cand_p5[0], atol=0.0001)) & (
                                                                 np.isclose(r_trj['y'], cand_p5[1],
                                                                            atol=0.0001))].values
                                found_id = trj_matched_data[0][1]  # 1 is tid
                                list_m_ids.append(found_id)
                        found_id = max(set(list_m_ids), key=list_m_ids.count)
                        print("Matched ids: " + str(list_m_ids) + " Found id: " + str(found_id) +
                              " Succsess/fail: " + str(points_dif) + " Current id: " + str(cur_tid_num))
                        # print("Found ids: id1" + str(found_id))
                        nxt_r_trj = r_trj.loc[
                            (r_trj['frame'] == cur_frame_num + 1) & (r_trj['tid'] == found_id)]
                        # print("Found next point: " + str(nxt_r_trj))
                        s_df_4 = df_2d.loc[
                            (df_2d['frame'] == cur_frame_num + 1) & (df_2d['cid'] == which_camera.id)]
                        nxt_p = np.array([nxt_r_trj['x'], nxt_r_trj['y']])
                        cand_p6, pre_min2, _, _, _ = find_candidate(s_df_4, which_camera, s_camera, nxt_p,
                                                                    10, 50, True)
                        # print("Matching point: " + str(cand_p6))
                        # pre_c = cand_p6

                        r_trj = pd.read_csv(path_to_check_trj)
                        nxt_r_trj = r_trj.loc[
                            (r_trj['frame'] == k - 1) & (r_trj['tid'] == found_id)]
                        # print("Found next point: " + str(nxt_r_trj))
                        s_df_4 = df_2d.loc[
                            (df_2d['frame'] == k - 1) & (df_2d['cid'] == which_camera.id)]
                        s_df_5 = df_2d.loc[
                            (df_2d['frame'] == k - 1) & (df_2d['cid'] == s_camera.id)]
                        nxt_p = np.array([nxt_r_trj['x'], nxt_r_trj['y']])
                        cand_p6, pre_min2, _, _, _ = find_candidate(s_df_4, which_camera, s_camera, nxt_p,
                                                                    10, 50, True)
                        cand_p7, pre_min2, _, _, _ = find_candidate(s_df_5, s_camera, which_camera, cand_p6,
                                                                    10, 50, True)
                        if len(cand_p6) != 0 and len(cand_p7) != 0:
                            cp_check = distance.euclidean(cp, cand_p6)
                            print("Cp check was: " + str(cp_check))
                        for nf in range(2):
                            nxt_r_trj = r_trj.loc[
                                (r_trj['frame'] == k + nf + 1) & (r_trj['tid'] == found_id)]
                            # print("Found next point: " + str(nxt_r_trj))
                            s_df_4 = df_2d.loc[
                                (df_2d['frame'] == k + nf + 1) & (df_2d['cid'] == which_camera.id)]
                            s_df_5 = df_2d.loc[
                                (df_2d['frame'] == k + nf + 1) & (df_2d['cid'] == s_camera.id)]
                            nxt_p = np.array([nxt_r_trj['x'], nxt_r_trj['y']])
                            cand_p6, pre_min2, _, _, _ = find_candidate(s_df_4, which_camera, s_camera, nxt_p,
                                                                        10, 50, True)
                            cand_p7, pre_min2, _, _, _ = find_candidate(s_df_5, s_camera, which_camera, cand_p6,
                                                                        10, 50, True)
                            cand_p8, pre_min2, _, _, _ = find_candidate(s_df_4, which_camera, s_camera, cand_p7,
                                                                        10, 50, True)
                            # print("Matching point2: " + str(cand_p6))
                            next_c_p = cand_p6
                            if len(next_c_p) != 0:
                                ind2 = tmp_2d_c1.loc[(tmp_2d_c1['frame'] == cur_frame_num + 2 + nf) &
                                                     (np.isclose(tmp_2d_c1['x'], next_c_p[0], atol=0.0001)) & (
                                                         np.isclose(tmp_2d_c1['y'], next_c_p[1], atol=0.0001))].index
                            # dist3 = distance.euclidean(next_c_p - pre_c)
                            if len(cand_p7) != 0 and len(cand_p6) != 0 and len(cand_p8) != 0:
                                dist3ch = distance.euclidean(nxt_p, cand_p7)
                                dist4pc = distance.euclidean(cand_p8, cand_p6)
                                cp_check = distance.euclidean(cp, cand_p8) / (nf + 2)
                                print("Cp check was: " + str(cp_check / (nf + 2)))
                                print("Pre min was: " + str(pre_min2))

                            if pre_min2 < 20.0 and dist3ch < 0.001 and dist4pc < 0.001 and points_dif < 1.0 and len(
                                    next_c_p) != 0:
                                print("Following frame is matched. Setting frame: " + str(k + nf + 1))
                                tmp_2d_c1.loc[ind2, 'matched'] = cur_tid_num
                                # mtp1 = tmp_2d_c1.loc[ind2, 'x'].values
                                # mtp2 = tmp_2d_c1.loc[ind2, 'y'].values
                                new_row = pd.DataFrame({"frame": [k + nf + 1], "tid": [cur_tid_num],
                                                        "x": next_c_p[0], "y": next_c_p[1], "matched": [cur_tid_num]})

                                new_row.to_csv(path_to_check_1, index=False, mode='a', header=False)
                            else:
                                print("Not matched enough.")
                                break
                        pass

            if pre_i != -1:
                trj_2d_df_match = pd.read_csv(path_to_check_1)
                ind = trj_2d_df_match.loc[
                    (trj_2d_df_match['frame'] == k - 1) & (trj_2d_df_match['x'] == pre_pc[0]) & (
                            trj_2d_df_match['y'] == pre_pc[1])].index
                ind2 = tmp_2d_c1.loc[(tmp_2d_c1['frame'] == k) &
                                     (np.isclose(tmp_2d_c1['x'], pre_c[0], atol=0.0001)) & (
                                         np.isclose(tmp_2d_c1['y'], pre_c[1], atol=0.0001))].index

                trj_2d_df_match.loc[ind, 'matched'] = cur_tid_num
                tmp_2d_c1.loc[ind2, 'matched'] = cur_tid_num
                new_row = pd.DataFrame({"frame": [k], "tid": [cur_tid_num],
                                        "x": pre_c[0], "y": pre_c[1], "matched": [cur_tid_num]})

                new_row.to_csv(path_to_check_1, index=False, mode='a', header=False)
    if deep_check:
        print("Lowest distance is: " + str(lowest_dist))


def match_tra_with_other_camera(df1, df2):
    s_counter = 0
    f_counter = 0
    run_len = len(df1)
    perf = 0
    for i in range(run_len):
        if np.isclose(df1.iloc[i]['x'], df2.iloc[i]['x'], atol=0.00001):
            if np.isclose(df1.iloc[i]['y'], df2.iloc[i]['y'], atol=0.00001):
                s_counter += 1
            else:
                f_counter += 1
        else:
            f_counter += 1
    if s_counter + f_counter != 0:
        perf = s_counter / (s_counter + f_counter) * 100
        print("Trajectory performance: " + str(perf))
    if perf > 0.9:
        # TODO: Matched previous frames, 1) project previous 2 frames to 3D, 2) calculate depth with speed,
        # TODO: 3) project next 1 frame to other camera 4) closest match on other camera should matched to tid
        pass


def compare_2d_tra_to_gt(tra_id, oid, which_camera: classes.Camera, file_name):
    path_to_check_1 = Path("gt_files/proj_2d.csv")
    path_to_check_2 = Path(f"result_files/{file_name}.csv")
    scounter = 0
    fcounter = 0
    if path_to_check_1.exists() and path_to_check_2.exists():
        df_2d = pd.read_csv(path_to_check_1)
        df_2d_tra = pd.read_csv(path_to_check_2)
        df_2d = df_2d.sort_values('frame')
        df_2d = df_2d.loc[df_2d['cid'] == which_camera.id]
        df_2d = df_2d.loc[df_2d['oid'] == oid]
        df_2d_tra = df_2d_tra.loc[df_2d_tra['tid'] == tra_id]
        len_2d_gt = len(df_2d)
        len_2d_tra = len(df_2d_tra)

        if len_2d_gt == len_2d_tra:
            run_len = len_2d_gt
        elif len_2d_gt > len_2d_tra:
            run_len = len_2d_tra
        else:
            run_len = len_2d_gt
        for i in range(run_len):
            if np.isclose(df_2d_tra.iloc[i]['x'], df_2d.iloc[i]['x'], atol=0.0001):
                if np.isclose(df_2d_tra.iloc[i]['y'], df_2d.iloc[i]['y'], atol=0.0001):
                    scounter += 1
                else:
                    fcounter += 1
            else:
                fcounter += 1
    if scounter + fcounter != 0:
        perf = scounter / (scounter + fcounter) * 100
        print("Trajectory performance: " + str(perf))


def calc_dist_2d_line_to_point(p1, p2, p3):
    return np.abs(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def distance_to_line(p1, p2, self):
    x_diff = p2[0] - p1[0]
    y_diff = p2[1] - p1[1]
    num = abs(y_diff * self[0] - x_diff * self[1] + p2[0] * p1[1] - p2[1] * p1[0])
    den = math.sqrt(y_diff ** 2 + x_diff ** 2)
    return num / den


def run_setup():
    '''
    Generates synthetic cameras and points. 
    Two cameras are generated with hard-coded parameters. When points are generated, 
    they will always be in FOV of both cameras (unless there's a lot of 3D noise.).
    
    
    Parameters
    ----------
    None 
    
    Returns 
    -------
    camera1, camera2, camera3 : Camera instances
    
    
    See Also 
    --------
    classes.Camera
    
    '''
    # Camera setup:
    f = 1230
    c_x = 960.
    c_y = 540.
    f_x = f
    f_y = f

    # Convert f to mm: focal length
    # F(mm) = F(pixels) * SensorWidth(mm) / ImageWidth(pixel).
    # GoPro Hero3 White Sensor Dimensions: 5.76 mm x 4.29 @ 1920x1080 & f = 1230
    camera_focal_len_mm = f * 0.00576 / 1920

    # Camera 1's intrinsic matrix.
    # Sensor width:	15.34 mm (1920 pixels)
    # Sensor height: 8.63 mm (1080 pixels)
    # Optical format: 1.1" (28mm), pixel binning 2x2, focal length: 6mm
    camera_im = [[f_x, 0.0, c_x], [0.0, f_y, c_y], [0.0, 0.0, 1.0]]
    camera_im = numpy.float32(camera_im)

    # Rotation matrices

    # 60 degrees towards Camera 2.
    camera_1_r_mtrx = [[0.98480775301, 0.0, -0.17364817766], [0.0, 1.0, 0.0],
                       [-0.17364817766 * - 1.0, 0.0, 0.98480775301]]
    camera_1_r_mtrx = np.float32(camera_1_r_mtrx)

    # 60 degrees towards Camera 1.
    camera_2_r_mtrx = [[0.98480775301, 0.0, 0.17364817766], [0.0, 1.0, 0.0],
                       [0.17364817766 * - 1.0, 0.0, 0.98480775301]]
    camera_2_r_mtrx = np.float32(camera_2_r_mtrx)

    # 90 degrees towards Camera 1.
    camera_3_r_mtrx = [[0, 0.0, 1], [0.0, 1.0, 0.0], [-1.0 * 1.0, 0.0, 0]]
    camera_3_r_mtrx = np.float32(camera_3_r_mtrx)

    # Translation matrix
    camera_1_t_mtrx = [0.0, 0.0, 0.0]
    camera_1_t_mtrx = np.float32(camera_1_t_mtrx)

    camera_2_t_mtrx = [-2.0, 0.0, 0.0]
    camera_2_t_mtrx = np.float32(camera_2_t_mtrx)

    camera_3_t_mtrx = [-10, 0.0, 10.0]
    camera_3_t_mtrx = np.float32(camera_3_t_mtrx)

    # Distortion coefficients matrix
    # cof_m = [-0.32, 0.126, 0, 0, 0]
    cof_m = [0.0, 0.0, 0, 0, 0]
    cof_m = np.float32(cof_m)

    # Cameras relative positions.
    tm3 = [20, 0.0, 0.0]
    tm3 = np.float32(tm3)
    tm4 = [-20, 0.0, 0.0]
    tm4 = np.float32(tm4)

    # Calculate epipolar points.
    rvet, _ = cv2.Rodrigues(camera_1_r_mtrx)
    rvet_2, _ = cv2.Rodrigues(camera_2_r_mtrx)
    rvet_3, _ = cv2.Rodrigues(camera_3_r_mtrx)

    oi, _ = cv2.projectPoints(tm3, rvet, camera_1_t_mtrx, camera_im, cof_m)
    oi_2, _ = cv2.projectPoints(tm4, rvet_2, camera_2_t_mtrx, camera_im, cof_m)
    oi_3, _ = cv2.projectPoints(camera_1_t_mtrx, rvet_3, camera_3_t_mtrx, camera_im, cof_m)

    tm3 = [25.0, 0.0, 0.0]
    tm3 = np.float32(tm3)
    tm4 = [-25.0, 0.0, 0.0]
    tm4 = np.float32(tm4)
    rel_pos_1 = [0, 0.0, 0.0]
    rel_pos_2 = [2.0, 0.0, 0.0]
    rel_pos_3 = [10.0, 0.0, 10.0]

    # 3x4 Camera Matrix
    cm_1 = [[camera_1_r_mtrx[0][0], camera_1_r_mtrx[0][1], camera_1_r_mtrx[0][2], camera_1_t_mtrx[0]],
            [camera_1_r_mtrx[1][0], camera_1_r_mtrx[1][1], camera_1_r_mtrx[1][2], camera_1_t_mtrx[1]],
            [camera_1_r_mtrx[2][0], camera_1_r_mtrx[2][1], camera_1_r_mtrx[2][2], camera_1_t_mtrx[2]]]
    cm_1 = np.matmul(camera_im, cm_1)
    cm_1 = numpy.float32(cm_1)

    cm_2 = [[camera_2_r_mtrx[0][0], camera_2_r_mtrx[0][1], camera_2_r_mtrx[0][2], camera_2_t_mtrx[0]],
            [camera_2_r_mtrx[1][0], camera_2_r_mtrx[1][1], camera_2_r_mtrx[1][2], camera_2_t_mtrx[1]],
            [camera_2_r_mtrx[2][0], camera_2_r_mtrx[2][1], camera_2_r_mtrx[2][2], camera_2_t_mtrx[2]]]
    cm_2 = np.matmul(camera_im, cm_2)
    cm_2 = numpy.float32(cm_2)

    cm_3 = [[camera_3_r_mtrx[0][0], camera_3_r_mtrx[0][1], camera_3_r_mtrx[0][2], camera_3_t_mtrx[0]],
            [camera_3_r_mtrx[1][0], camera_3_r_mtrx[1][1], camera_3_r_mtrx[1][2], camera_3_t_mtrx[1]],
            [camera_3_r_mtrx[2][0], camera_3_r_mtrx[2][1], camera_3_r_mtrx[2][2], camera_3_t_mtrx[2]]]
    cm_3 = np.matmul(camera_im, cm_3)
    cm_3 = numpy.float32(cm_3)

    camera_1 = classes.Camera(0, oi, f, c_x, c_y, f_x, f_y,
                              camera_im, camera_1_t_mtrx, camera_1_r_mtrx, cof_m, rel_pos_1, cm_1)
    camera_2 = classes.Camera(1, oi_2, f, c_x, c_y, f_x, f_y,
                              camera_im, camera_2_t_mtrx, camera_2_r_mtrx, cof_m, rel_pos_2, cm_2)
    camera_3 = classes.Camera(2, oi_3, f, c_x, c_y, f_x, f_y,
                              camera_im, camera_3_t_mtrx, camera_3_r_mtrx, cof_m, rel_pos_3, cm_3)
    return camera_1, camera_2, camera_3


def run_setup_starling():
    '''
    Parameters
    ----------
    None
    But - hard coded are 1) .xcp camera parameters file and 2) the unit conversion from 
    mm to m (/1000)

    Returns
    -------
    camera_df : pd.DataFrame
        With columns : 
            did : device id
            focal_length
            o1,o2,o3,o4 : orientations of camera
            x,y,z 
            ppx,ppy,ppz : principal point x,y,z
            coe1-5 : distortion coeffieicnts
    '''

    # Camera setups
    xmldoc = minidom.parse('gt_files/cameras_starling.xcp')
    itemlist = xmldoc.getElementsByTagName('Camera')
    # print(len(itemlist))
    # print(itemlist[0].attributes['DEVICEID'].value)
    itemlist2 = xmldoc.getElementsByTagName('KeyFrame')
    # print(itemlist2[0].attributes['FOCAL_LENGTH'].value)

    # did: Device id; o1...o4: Orientation; x,y,z: Position; ppx,ppy: Principal Point;
    # coe1...coew5: ViconRadial2(Distortion Coefficients)
    camera_df = pd.DataFrame(
        columns=[
            "did", "focal_length", "o1", "o2", "o3", "o4", "x", "y", "z",
            "ppx", "ppy", "coe1", "coe2", "coe3", "coe4", "coe5"
        ])

    counter = 0
    for s in itemlist:
        fls = itemlist2[counter].attributes['FOCAL_LENGTH'].value
        orientation = itemlist2[counter].attributes['ORIENTATION'].value
        orientation_list = orientation.split()
        pos3d = itemlist2[counter].attributes['POSITION'].value
        pos3d_list = pos3d.split()
        pr_point = itemlist2[counter].attributes['PRINCIPAL_POINT'].value
        pr_point_list = pr_point.split()
        coff = itemlist2[counter].attributes['VICON_RADIAL2'].value
        coff_list = coff.split()
        did = s.attributes['DEVICEID'].value

        camera_df = camera_df.append(
            pd.DataFrame({"did": int(did), "focal_length": numpy.float32(fls), "o1": numpy.float32(orientation_list[0]),
                          "o2": numpy.float32(orientation_list[1]), "o3": numpy.float32(orientation_list[2]),
                          "o4": numpy.float32(orientation_list[3]), "x": numpy.float32(pos3d_list[0]) / 1000.0,
                          "y": numpy.float32(pos3d_list[1]) / 1000.0, "z": numpy.float32(pos3d_list[2]) / 1000.0,
                          "ppx": numpy.float32(pr_point_list[0]), "ppy": numpy.float32(pr_point_list[1]),
                          "coe1": numpy.float32(coff_list[1]), "coe2": numpy.float32(coff_list[2]),
                          "coe3": numpy.float32(coff_list[3]), "coe4": numpy.float32(coff_list[4]),
                          "coe5": numpy.float32(coff_list[5])}, index=[counter]))

        counter += 1
    # print(camera_df)
    return camera_df


def create_camera(camera_df, index):
    '''
    Creates camera instances from a dataframe.
    
    Parameters
    ----------
    camera_df : pd.DataFrame 
        with columns of focal_length, ppx, ppy, etc. 
    index : int 
        Camera number 
    
    Returns 
    -------
    camera_1 : classes.Camera instance
    '''
    # Camera setup:
    f = camera_df.iloc[index]['focal_length']
    c_x = camera_df.iloc[index]['ppx']
    c_y = camera_df.iloc[index]['ppy']
    f_x = f
    f_y = f
    t_mtrx = [camera_df.iloc[index]['x'], camera_df.iloc[index]['y'], camera_df.iloc[index]['z']]
    # camera_df.iloc[index]['x'], camera_df.iloc[index]['y'], camera_df.iloc[index]['z']
    rel_pos_1 = [camera_df.iloc[index]['x'] * -1.0, camera_df.iloc[index]['y'] * -1.0,
                 camera_df.iloc[index]['z'] * -1.0]

    # Camera's intrinsic matrix
    camera_im = [[f_x, 0.0, c_x], [0.0, f_y, c_y], [0.0, 0.0, 1.0]]
    camera_im = numpy.float32(camera_im)

    # Distortion coefficients matrix
    cof_m = [camera_df.iloc[index]['coe3'], camera_df.iloc[index]['coe4'], camera_df.iloc[index]['coe5'], 0.0, 0.0]
    cof_m = np.float32(cof_m)

    # Converting 1x4 orientation (rotation) matrix to 3x3
    divd = math.sqrt(1 - camera_df.iloc[index]['o4'] * camera_df.iloc[index]['o4'])
    om_1 = [[camera_df.iloc[index]['o1'] / divd],
            [camera_df.iloc[index]['o2'] / divd],
            [camera_df.iloc[index]['o3'] / divd]]
    om_1 = numpy.float32(om_1)
    om_converted, _ = cv2.Rodrigues(om_1)

    # 3x4 Camera Matrix
    cm_1 = [[om_converted[0][0], om_converted[0][1], om_converted[0][2], t_mtrx[0]],
            [om_converted[1][0], om_converted[1][1], om_converted[1][2], t_mtrx[1]],
            [om_converted[2][0], om_converted[2][1], om_converted[2][2], t_mtrx[2]]]
    cm_1 = np.matmul(camera_im, cm_1)
    cm_1 = numpy.float32(cm_1)
    camera_1 = classes.Camera(index, camera_df.iloc[index]['did'], f, c_x, c_y, f_x, f_y,
                              camera_im, t_mtrx, om_converted, cof_m, rel_pos_1, cm_1)
    return camera_1


def find_best_pair_of_cameras(camera_df, gt_3d_df):
    '''
    Projects 3D groundtruth data onto each camera, and checks the number
    of points with valid 2D projections. 
    
    Parameters
    ----------
    camera_df : pd.DataFrame
        Output from run_setup_starling()

    See Also
    --------
    run_setup_starling
    '''
    best_c1, best_c2 = math.inf, math.inf
    best_c1_id, best_c2_id = -1, -1
    for i in range(len(camera_df)):
        c1 = create_camera(camera_df, i)
        _, fail_counter = data_functions.project_to_2d_and_3d(gt_3d_df, c1, mu=0.0, sigma=0.0, mu3d=0, sigma3d=0)
        if fail_counter < best_c2:
            if fail_counter < best_c1:
                best_c1 = fail_counter
                best_c1_id = camera_df.iloc[i]['did']
            else:
                best_c2 = fail_counter
                best_c2_id = camera_df.iloc[i]['did']
        if fail_counter <= 0:
            print("Working camera id: " + str(camera_df.iloc[i]['did']))
        else:
            print("Failed camera id: " + str(camera_df.iloc[i]['did']))
    print("Found: " + str(best_c1_id) + ", " + str(best_c2_id))
    print("")


if __name__ == '__main__':
    print("Running")
    # test_case()

    # kalman_setup()
    # kfilter()

    # kalman_fill("recon1", 10, 6001)

    camera_df = run_setup_starling()

    camera_1, camera_2, camera_3 = run_setup()

    # kalman_pair("recon2", "proj_2d", camera_2)

    if force_calc and create_objects:
        # Creates a csv file with 3D ground truth trajectories
        data_functions.create_data_3d_object_data_2(how_many_objects_to_create, 30, 20)
        pass

    gt_3d_df, r1 = data_functions.read_data_file("gt_3d_data_v1")
    starling_df, starling_r1 = data_functions.read_starling_data_file("ST01a.vsk")

    gt_3d_df = starling_df
    # ST06, 07, 08
    # found_camera_df = camera_df.loc[(camera_df['did'] == 2121154) | (camera_df['did'] == 2121113)]
    # ST01, 02, 03
    # found_camera_df = camera_df.loc[(camera_df['did'] == 2121156) | (camera_df['did'] == 2122725)]
    # ST09, 10, 11
    # found_camera_df = camera_df.loc[(camera_df['did'] == 2121113) | (camera_df['did'] == 2121156)]
    # ST 12a, 13b, 14b (Starling_Trials_07-12-2019_11-15-00RESULT)
    found_camera_df = camera_df.loc[(camera_df['did'] == 2121113) | (camera_df['did'] == 2121154)]

    # Fundamental matrix calculation
    # fmatrix = calcFundamentalMatrix(camera_1, camera_2) # for synthetic data
    # for the startling data (below)
    fmatrix = calcFundamentalMatrix(create_camera(found_camera_df, 0), create_camera(found_camera_df, 1))

    # find_best_pair_of_cameras(camera_df, gt_3d_df)

    print('...found best pair')
    if r1 and do_estimations:
        # Get projection dataframes.
        # _, _ = data_functions.project_to_2d_and_3d(gt_3d_df, camera_1, mu=0.0, sigma=0.0, mu3d=0, sigma3d=0)
        # _, _ = data_functions.project_to_2d_and_3d(gt_3d_df, camera_2, mu=0.0, sigma=0.0, mu3d=0, sigma3d=0)
        _, _ = data_functions.project_to_2d_and_3d(gt_3d_df, create_camera(found_camera_df, 0),
                                                   mu=0.0, sigma=0.0, mu3d=0, sigma3d=0)
        _, _ = data_functions.project_to_2d_and_3d(gt_3d_df, create_camera(found_camera_df, 1),
                                                   mu=0.0, sigma=0.0, mu3d=0, sigma3d=0)
        object_2d_df, r2 = data_functions.read_data_file("proj_2d")
        if gt_construction and r2:
            reconstruct_trajectories(create_camera(found_camera_df, 0), create_camera(found_camera_df, 1),
                                     object_2d_df, gt_3d_df, "reconstruction")
            # reconstruct_trajectories(camera_1, camera_2, object_2d_df, gt_3d_df, "reconstruction")
        elif r2:
            estimate_3d_points(create_camera(found_camera_df, 0), create_camera(found_camera_df, 1),
                               object_2d_df, gt_3d_df, "recon2", fmatrix)
            # estimate_3d_points(camera_1, camera_2, object_2d_df, gt_3d_df, "recon2", fmatrix)
            # find_trajectory_2d(object_2d_df, camera_2, 2, 5, "trajectories2d", 0, True)
            # find_trajectory_2d(object_2d_df, camera_1, 2, 999, "trajectories2d", 0, True, False)
            # match_2d_trajectories(camera_1, camera_2, "recon1", "recon2", "matched2d", gt_3d_df)
            pass

        # vis_3d_3(True)
    # compare_2d_tra_to_gt(1, 0, create_camera(found_camera_df, 0), "trajectories2d")
    # compare_2d_tra_to_gt(0, 0, create_camera(found_camera_df, 0), "trajectories2d")
    # compare_2d_tra_to_gt(1, 1, create_camera(found_camera_df, 0), "trajectories2d")

    if visualization:
        visualizations.vis_3d(False, 1)
        # visualizations.vis_2d_tid(False, "tids", camera_1, 0, 2)
        # visualizations.vis_2d(False, "out_tr1d", camera_1, False)
        # visualizations.vis_2d(True, "gt_tr1d", camera_1, False)
        # visualizations.vis_2d(False, "out_tr2d_no_geo", camera_2, True)
        pass

    print("End")
