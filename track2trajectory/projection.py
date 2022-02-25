'''
Projection
==========
Module which handles 2D and 3D projection and associated functions

'''
import numpy as np 
import cv2
import pandas as pd
import track2trajectory.camera as camera

def calcFundamentalMatrix(camera_1, camera_2):
    '''
    Calculates the fundamental matrix that transforms 2D points on camera 1
    to provide the corresponding 2D points on camera 2

    Uses the formula :math:`F = [e']_{x} P' P^{+}`

    Where:

    * :math:`[e']_{x}` is the skew-symmetric matrix from e
    * :math:`e'` is the epipole, obtained by :math:`e' = P' C`, where :math:`C` is the camera1 centre
    * :math:`P'` is the projection matrix of camera 2
    * :math:`P^{+}` is the pseudo-inverse of camera 1's projection matrix

    Parameters
    ----------
    camera_1, camera_2 : camera.Camera instances 

    Returns
    -------
    F : 3x3 np.array 
        2D array with the fundamental matrix. 

    Notes
    -----                      
    The fundamental matrix is a 3X3 matrix which maps a point on cam1 onto the
    corresponding point on cam2

    References
    ----------
    * pg. 246 and pg 581 of Hartley & Zisserman 2003

    See Also
    --------
    make_skew_symmetric_matrix
    '''
    if not camera_1.id < camera_2.id:
        ValueError(f'Camera 1 ID: {camera_1.id} is > Camera 2 ID: {camera_2.id}.\
                   For calcFundamentalMatrix Camera 1 ID must be < Camera 2 ID')
    # M is defined on pg 161 as 'the first 3x3 submatrix of P'
    M = camera_1.cm_mtrx[:3,:3]
    p4 = camera_1.cm_mtrx[:,-1]
    # Calculating camera centre C on page 158
    C_inhomog = -np.matmul(np.linalg.inv(M),p4)
    C = np.concatenate((C_inhomog, np.array([1]))).flatten() # make homogeneous

    P_prime = camera_2.cm_mtrx
    e_prime = np.matmul(P_prime, C)
    e_prime_cross = make_skew_symmetric_matrix(e_prime)

    # P+ is the pseudo inverse of P 
    P = camera_1.cm_mtrx

    # Check that PC = 0 
    if not np.allclose(np.matmul(P,C), [0,0,0],atol=1e-6):
        raise ValueError('PC condition not satisfied.\
                         Check camera_1 projection matrix...')

    P_plus = np.linalg.pinv(P)
    Pprime_Pplus = np.matmul(P_prime, P_plus)
    F = np.matmul(e_prime_cross, Pprime_Pplus)
    return F

def project_to_2d_and_3d(gt_3d_df, camera_obj: camera.Camera,
                         mu=0, sigma=0, mu3d=0, sigma3d=0):
    '''
    Projects 2D points from the input 3D trajectories while also optionally 
    adding noise. 

    Two types of noise are implemented (both can be also used). Pre-projection noise
    adds 3D noise to the xyz coordinates, and post-projection nosie adds 2D noise 
    to the pixel coordinates. 
    
    Parameters
    ----------
    gt_3d_df : (Mrows, Ncols) pd.DataFrame 
        With 3D trajectory information and columns frame, oid, and x,y,z
    camera_obj : classes.Camera instance
    mu : float, optional
        2D noise mean (noise post projection). Default 0
    sigma : float, optional
        2D noise standard deviation (post projection). Default 0
    mu3d : float, optional
        3D noise mean pre-projection. Default 0
        The same mean is applied to x,y,z.
        i.e. x + N(mu3d, sigma3d), y+ N(mu3d,sigma3d), z+ N(mu3d, sigma3d)
    sigma3D : float, optional
        3D noise standard deviation pre-projection. Default 0
        The same sigma is applied to x,y,z

    Returns
    -------
    proj_2d_df : (Mrows, 5) pd.DataFrame
        2D projection dataframe with columns:
            frame
            oid (object id)
            cid (camera id)
            x : column number (increases to the right)
            y : the row number (increases as you move down the image)
        If projection of a point isnt possible onto the camera view, then NaN
        is returned.

    fail_counter : int
        Number of positions that can't be projected onto the 2D image.
    '''
    proj_2d_df = pd.DataFrame()
    fail_counter = 0

    for i in range(len(gt_3d_df)):
        # 3D noise
        noise3d = np.random.normal(mu3d, sigma3d, [3, 1])
        # HERE THE Y AND Z COORDINATES NEED TO BE SWITCHED FOR THE SYSTEM TO 
        # MAKE SENSE.
        # Set values & add noise. 
        x_temp = gt_3d_df.iloc[i]['x'] + noise3d[0]
        y_temp = gt_3d_df.iloc[i]['z'] + noise3d[2]
        z_temp = gt_3d_df.iloc[i]['y'] + noise3d[1]
        frame_temp = gt_3d_df.iloc[i]['frame']

        oid = gt_3d_df.iloc[i]['oid']
        cid = camera_obj.id
        obj_xyz_temp = np.float32([x_temp, y_temp, z_temp])
        # Project Points from 3D to 2D
        r_m, _ = cv2.Rodrigues(camera_obj.r_mtrx)
        oi, _ = cv2.projectPoints(obj_xyz_temp, r_m,
                                  camera_obj.t_mtrx, 
                                  camera_obj.i_mtrx, 
                                  camera_obj.cof_mtrx)
        twodpoint_in_xplane = camera_obj.c_x*2 >= oi[0][0][0] >= 0.0
        twodpoint_in_yplane = 0.0 <= oi[0][0][1] <= camera_obj.c_y*2
        # Check if object is in image plane
        if twodpoint_in_xplane and twodpoint_in_yplane:
            # Set 2D points to data frame.
            # Gaussian noise
            noise = np.random.normal(mu, sigma, [2, 1])
            x2d = np.float32(noise[0] + oi[0][0][0])
            y2d = np.float32(noise[1] + oi[0][0][1])
        else:
            # if projection is beyond the sensor - then NaN
            fail_counter +=1
            x2d = np.nan
            y2d = np.nan
        
        proj_2d_df = proj_2d_df.append(pd.DataFrame({"frame": [frame_temp],
                                                         "oid": [oid],
                                                         "x": x2d, "y": y2d,
                                                         "cid": cid}))

    return proj_2d_df.reset_index(drop=True), fail_counter

def triangulate_points_in3D(point1, point2, camera1, camera2):
    '''
    Parameters
    ----------
    point1, point2 : (2,) np.array with np.float32 entries
    camera1, camera2 : camera.Camera instances
    
    Returns
    -------
    xyz_world : (3,) np.array with np.float32 entries
    '''
    tri_res = cv2.triangulatePoints(camera1.cm_mtrx, camera2.cm_mtrx,
                                    point1.reshape(-1,1), point2.reshape(-1,1))

    xyz_camera = cv2.convertPointsFromHomogeneous(tri_res.T).flatten()
    # camera axis has Z pointing out, now we need to make it into Y pointing out
    xyz_world = np.float32([xyz_camera[0], xyz_camera[2], xyz_camera[1]])
    return xyz_world

def make_skew_symmetric_matrix(a):
    '''

    if a is a vector with a = (a1, a2, a3)
    then the skew-matrix of a, called :math:`[a]_{x}`
    is:
    
    ::
    
         | 0  |  -a3| a2 |
         | a3 |   0 | -a1|
         |-a2 |  a1 |  0 |

    Parameters
    ----------
    a : (3,) np.array, list-like
    
    Returns
    -------
    a_x : (3,3) np.array
        The skew-symmetric matrix

    References
    ----------
    * Appendix 4.2, page 581 of Hartley & Zisserman 2003 for :math:`[a]_{x}`
    '''
    a1,a2,a3 = a
    a_x = np.array(([0, -a3, a2],
                    [a3, 0, -a1],
                    [-a2, a1, 0]), dtype=np.float32)
    return a_x