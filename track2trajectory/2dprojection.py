import numpy as np 

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

