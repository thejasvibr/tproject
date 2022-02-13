import math
import numpy as np 
import cv2
import pandas as pd
import track2trajectory.camera as camera


def calcFundamentalMatrix(camera_1_o, camera_2_o):
    '''
    Calculates the fundamental matrix 

    Parameters
    ----------
    camera_1_o, camera_2_o : camera.Camera instances 
    
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


def project_to_2d_and_3d(gt_3d_df, camera_obj: camera.Camera,
                         mu=0, sigma=0, mu3d=0, sigma3d=0):
    '''
    Projects 2D points from the input 3D trajectories while also optionally 
    adding noise. 
    
    Two types of noise are implemented (both can be also used). Pre-projection noise
    adds 3D noise to the xyz coordinates, and post-projection nosie adds 2D noise 
    to the pixel coordinates. 
    
    The projection is always expected to be within 1920 x 1080 pixels
    
    Parameters
    ----------
    gt_3d_df : pd.DataFrame 
        With 3D trajectory information and columns frame, oid, and x,y,z
    camera_obj : classes.Camera instance
    mu : float, optional
        2D noise mean (noise post projection). Default 0
    sigma : float, optional
        2D noise standard deviation (post projection). Default 0
    mu3d : float, optional
        3D noise mean pre-projection. Default 0
    sigma3D : float, optional
        3D noise standard deviation pre-projection. Default 0
    
    Returns
    -------
    proj_2d_df : pd.DataFrame
        2D projection dataframe with columns:
            frame
            oid (object id)
            cid (camera id)
            x : column number (increases to the right)
            y : the row number (increases as you move down the image)

    fail_counter : int
        Number of positions that can't be projected onto the 2D image.    
   
    '''
    proj_2d_df = pd.DataFrame()
    fail_counter = 0

    for i in range(len(gt_3d_df)):
        # 3D noise
        noise3d = np.random.normal(mu3d, sigma3d, [3, 1])
        # Set values & add noise. Divided by 60 due to distribution to achieve up to 10cm+ noise.
        x_temp = gt_3d_df.iloc[i]['x'] + noise3d[0] / 120.0
        y_temp = gt_3d_df.iloc[i]['y'] + noise3d[1] / 120.0
        z_temp = gt_3d_df.iloc[i]['z'] + noise3d[2] / 120.0
        frame_temp = gt_3d_df.iloc[i]['frame']

        oid = gt_3d_df.iloc[i]['oid']
        cid = camera_obj.id
        obj_xyz_temp = [x_temp, y_temp, z_temp]
        obj_xyz_temp = np.float32(obj_xyz_temp)

        # Project Points from 3D to 2D
        r_m, _ = cv2.Rodrigues(camera_obj.r_mtrx)
        oi, _ = cv2.projectPoints(obj_xyz_temp, r_m,
                                  camera_obj.t_mtrx, camera_obj.i_mtrx, camera_obj.cof_mtrx)

        # Check if object is in image plane
        if camera_obj.c_x*2 >= oi[0][0][0] >= 0.0 and 0.0 <= oi[0][0][1] <= camera_obj.c_y*2:
            # Set 2D points to data frame.
            # Gaussian noise
            noise = np.random.normal(mu, sigma, [2, 1])
            x2d = np.float32(noise[0] + oi[0][0][0])
            y2d = np.float32(noise[1] + oi[0][0][1])
            proj_2d_df = proj_2d_df.append(pd.DataFrame({"frame": [frame_temp], "oid": [oid],
                                                         "x": x2d, "y": y2d, "cid": cid}))

        else:
            fail_counter += 1

    print("Failed total points: " + str(fail_counter))
    return proj_2d_df, fail_counter

def find_candidate(df_2d_c1, fund_matrix, camera1: camera.Camera,
                   camera2: camera.Camera, candidate_point,
                   known_depth_for_camera, threshold, rec):
    '''
    Gives the 2D point closest to the epipolar line projected from the reference camera.

    Parameters
    ----------
    df_2d_c1 : pd.DataFrame
        xy coordinates for on the camera 'reference' camera  (assigned to number 1)
    fund_matrix: 3x3 np.array
        The fundamental matrix. The fundamental matrix is a matrix which transforms
        a set of points Xi on one camera into their corresponding points Xj onto 
        another camera.
    camera1, camera2 : camera.Camera instances
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
    projection.calcFundamentalMatrix
    
    
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
        df1numpya = np.float64(df_2d_c1[['x', 'y']].to_numpy())
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
