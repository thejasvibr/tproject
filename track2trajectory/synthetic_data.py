'''
Synthetic data
==============
Module that contains functions to generate synthetic cameras, 3D points 
and their 2D projections

Also has helper functions which allow specifying the camera matrices from 
rotations and centre coordinates in the world system. 


References
----------
* Collins, Robert, Introduction to Computer Vision lecture notes (lecture 12)
* Simek, Kyle, https://ksimek.github.io/2012/08/22/extrinsic/ (blog post) accessed 14/2/2022

'''
import math
import pandas as pd
import cv2
import numpy as np 
import track2trajectory.camera as camera
from scipy.spatial.transform import Rotation

def generate_three_synthetic_cameras():
    '''
    Generates three synthetic cameras and places them in 3D space.
    Two cameras are generated with hard-coded parameters. When points are generated, 
    they will always be in FOV of both cameras (unless there's a lot of 3D noise.).


    Parameters
    ----------
    None 

    Returns
    -------
    camera1, camera2, camera3 : Camera instances


    TODO
    ----
    * Camera parameters are currently hard coded, need to be made flexible
    
    
    Example
    -------
    
    > import track2trajectory.synthetic_data as syndata
    > cam1, cam2, cam3 =  syndata.generate_three_synthetic_cameras()   
    

    See Also 
    --------
    camera.Camera
    
    History
    -------
    this function was formerly 'run_setup'
    '''
    # Camera setup:
    f = 1230 # pixels
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
    camera_im = [[f_x, 0.0, c_x],
                 [0.0, f_y, c_y],
                 [0.0, 0.0, 1.0]]
    camera_im = np.float32(camera_im)

    # Rotation matrices

    # 60 degrees towards Camera 2.
    camera_1_r_mtrx = [[0.98480775301, 0.0, -0.17364817766], 
                       [0.0,           1.0,           0.0],
                       [-0.17364817766 * - 1.0, 0.0, 0.98480775301]]
    camera_1_r_mtrx = np.float32(camera_1_r_mtrx)

    # 60 degrees towards Camera 1.
    camera_2_r_mtrx = [[0.98480775301, 0.0, 0.17364817766],
                       [0.0,           1.0, 0.0],
                       [0.17364817766 * - 1.0, 0.0, 0.98480775301]]
    camera_2_r_mtrx = np.float32(camera_2_r_mtrx)

    # 90 degrees towards Camera 1.
    camera_3_r_mtrx = [[0, 0.0, 1],
                       [0.0, 1.0, 0.0],
                       [-1.0 * 1.0, 0.0, 0]]
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
    
    # Attention:
    # the camera translations are assigned from tm for cams1,2 but from camera1_t_mtrx
    # for camera 3!!!
    oi, _ = cv2.projectPoints(tm3, rvet, camera_1_t_mtrx, camera_im, cof_m)
    oi_2, _ = cv2.projectPoints(tm4, rvet_2, camera_2_t_mtrx, camera_im, cof_m)
    oi_3, _ = cv2.projectPoints(camera_1_t_mtrx, rvet_3, camera_3_t_mtrx, camera_im, cof_m)


    rel_pos_1 = [0, 0.0, 0.0]
    rel_pos_2 = [2.0, 0.0, 0.0]
    rel_pos_3 = [10.0, 0.0, 10.0]

    # 3x4 Camera Matrix
    cm_1 = [[camera_1_r_mtrx[0][0], camera_1_r_mtrx[0][1], camera_1_r_mtrx[0][2], camera_1_t_mtrx[0]],
            [camera_1_r_mtrx[1][0], camera_1_r_mtrx[1][1], camera_1_r_mtrx[1][2], camera_1_t_mtrx[1]],
            [camera_1_r_mtrx[2][0], camera_1_r_mtrx[2][1], camera_1_r_mtrx[2][2], camera_1_t_mtrx[2]]]
    cm_1 = np.matmul(camera_im, cm_1)
    cm_1 = np.float32(cm_1)

    cm_2 = [[camera_2_r_mtrx[0][0], camera_2_r_mtrx[0][1], camera_2_r_mtrx[0][2], camera_2_t_mtrx[0]],
            [camera_2_r_mtrx[1][0], camera_2_r_mtrx[1][1], camera_2_r_mtrx[1][2], camera_2_t_mtrx[1]],
            [camera_2_r_mtrx[2][0], camera_2_r_mtrx[2][1], camera_2_r_mtrx[2][2], camera_2_t_mtrx[2]]]
    cm_2 = np.matmul(camera_im, cm_2)
    cm_2 = np.float32(cm_2)

    cm_3 = [[camera_3_r_mtrx[0][0], camera_3_r_mtrx[0][1], camera_3_r_mtrx[0][2], camera_3_t_mtrx[0]],
            [camera_3_r_mtrx[1][0], camera_3_r_mtrx[1][1], camera_3_r_mtrx[1][2], camera_3_t_mtrx[1]],
            [camera_3_r_mtrx[2][0], camera_3_r_mtrx[2][1], camera_3_r_mtrx[2][2], camera_3_t_mtrx[2]]]
    cm_3 = np.matmul(camera_im, cm_3)
    cm_3 = np.float32(cm_3)

    camera_1 = camera.Camera(0, oi, f, c_x, c_y, f_x, f_y,
                              camera_im, camera_1_t_mtrx, camera_1_r_mtrx, cof_m,
                              rel_pos_1, cm_1)
    camera_2 = camera.Camera(1, oi_2, f, c_x, c_y, f_x, f_y,
                              camera_im, camera_2_t_mtrx, camera_2_r_mtrx, cof_m,
                              rel_pos_2, cm_2)
    camera_3 = camera.Camera(2, oi_3, f, c_x, c_y, f_x, f_y,
                              camera_im, camera_3_t_mtrx, camera_3_r_mtrx, cof_m,
                              rel_pos_3, cm_3)
    return camera_1, camera_2, camera_3


def generate_two_synthetic_cameras_version2(**kwargs):
    '''
    Generates 2 cameras with completely specifiable parameters
    This approach relies on getting the camera pose and location
    in the WORLD system, and then generating the world->camera 
    rotation and translation matrices. 

    Parameters
    ----------
    cam1_C, cam2_C : (1,3) np.array, optional
        Camera centres in world coordinates. Defaults to [-1,0,0] (xyz)
        for Cam1 and [1,0,0] (xyz) for cam 2
    cam1_Rotation, cam2_Rotation : (3,3) np.arrays, optional
        Rotation matrices of the camera. Remember that the camera z axis is the 
        front/back axis when making the rotation matrix. Defaults to 
        no rotation, equivalent to np.eye(3)
    f_m : (2,) list-like, optional
        focal lengths in metres. Defaults to 0.00369 m
    pixel_size : list-like, optional
        Pixel size in metres. Defaults to 3e-6
    ppx : (2,) list-like, optional
        principal x points for cam 1 and cam2. Defaults to [960,960]
    ppy : (2,) list-like, optional
        Principal y points for cam1 and 2. Defaults to [540,540]
    distortion_coefs : (2,) list-like, optional
        Distortion coefficients. Defaults to [[0,0,0,0], [0,0,0,0]]

    Returns
    -------
    camera_1, camera_2 : camera.Camera instance
        Two camera instances with the input parameters.

   
    Notes
    -----
    The default camera parameters mimic that of 2 GoPro Hero camera centres at [-1,0,0]
    and [1,0,0] world XYZ coordinates. Both cameras have no distortion. See Parameters
    for further details on camera parameters

    Example
    -------
    
    > import track2trajectory.synthetic_data as syndata
    > cam1, cam2, cam3 =  syndata.generate_two_synthetic_cameras_version2()
    
    '''
    # Camera centres in World coordinates
    cam1_C = kwargs.get('cam1_C',np.array([-1,0,0]))
    cam2_C = kwargs.get('cam2_C',np.array([1,0,0]))

    # Camera 1 and 2 (3,3) rotation matrices. Remember xy are in image plane,
    # and z faces out of the plane.
    rot_cam1 = kwargs.get('cam1_Rotation', 
                          Rotation.from_euler('xyz', 
                                              [0,0,0], degrees=True).as_matrix())
    rot_cam2 = kwargs.get('cam2_Rotation', 
                          Rotation.from_euler('xyz',
                                              [0,0,0], degrees=True).as_matrix())

    # Get the homogeneous R matrix with rotation and translation 
    R_cam1 = make_rotation_mat_fromworld(rot_cam1, cam1_C)
    R_cam2 = make_rotation_mat_fromworld(rot_cam2, cam2_C)

    # split the R matrix into the component rotation and translations
    rotation_cam1, rotation_cam2  = R_cam1[:3,:3], R_cam2[:3,:3]
    transl_cam1, transl_cam2 = R_cam1[:3, -1], R_cam2[:3, -1]

    # Set focal length in pixels
    # Default settings are the GoPro settings
    # F(mm) = F(pixels) * SensorWidth(mm) / ImageWidth(pixel).
    # GoPro Hero3 White Sensor Dimensions: 5.76 mm x 4.29 @ 1920x1080 & f = 1230 pixels
    pixel_size = np.array(kwargs.get('pixel_size', [3e-6]*2))
    f_m = np.array(kwargs.get('f_m', 1230*pixel_size)) # focal length in pixels

    f_pixels = f_m/pixel_size

    # Principal points
    c_x_cams = kwargs.get('ppx', [960, 960])
    c_y_cams = kwargs.get('ppx', [540, 540])

    # distortion coefficients
    distortion_coefs = kwargs.get('distortion_coefs', [[0,0,0,0], [0,0,0,0]])
    
    # intrinsic matrices
    cam1_int_mat= [[f_pixels[0],      0.0   , c_x_cams[0]],
                   [0.0        , f_pixels[0], c_y_cams[0]],
                   [0.0,          0.0,            1.0]]
    cam2_int_mat= [[f_pixels[1],      0.0   , c_x_cams[1]],
                   [0.0        , f_pixels[1], c_y_cams[1]],
                   [0.0,              0.0,        1.0]]

    cam1_projection_mat = np.matmul(cam1_int_mat, R_cam1[:3,:])
    cam2_projection_mat = np.matmul(cam2_int_mat, R_cam2[:3,:])

    camera_1 = camera.Camera(0, [None, None], f_pixels[0], c_x_cams[0], c_y_cams[0],
                                 f_pixels[0], f_pixels[0],
                              cam1_int_mat, transl_cam1, rotation_cam1,
                              distortion_coefs[0],
                              [None]*3, cam1_projection_mat)
    camera_2 = camera.Camera(1, [None, None], f_pixels[1], c_x_cams[1], c_y_cams[1],
                                 f_pixels[1], f_pixels[1],
                              cam2_int_mat, transl_cam2, rotation_cam2,
                              distortion_coefs[1],
                              [None]*3, cam2_projection_mat)
    return camera_1, camera_2
    

def create_camera(camera_df, index):
    '''
    Creates camera instance from a dataframe.
    
    Parameters
    ----------
    camera_df : pd.DataFrame 
        with columns of: focal_length, ppx, ppy, x, y, z, c_x, c_y, c_z, 
        f_x, f_y, f_z, coe1, coe2, coe3, coe4, coe5, o1, o2, o3, o4,
        did
        camera_df can also have multiple camera, each in one row. 
        focal_length is the camera focal length in pixels, as is f_x and f_y.
        c_x, c_y are the principal points in x and y.
    index : int 
        Row number, which corresponds to camera number 

    Returns 
    -------
    camera_1 : camera.Camera instance
    '''
    # Camera setup:
    f = camera_df.iloc[index]['focal_length'] # pixels
    c_x = camera_df.iloc[index]['ppx']
    c_y = camera_df.iloc[index]['ppy']
    f_x = f
    f_y = f
    t_mtrx = [camera_df.iloc[index]['x'], camera_df.iloc[index]['y'], camera_df.iloc[index]['z']]
    # camera_df.iloc[index]['x'], camera_df.iloc[index]['y'], camera_df.iloc[index]['z']
    rel_pos_1 = [camera_df.iloc[index]['x'] * -1.0, camera_df.iloc[index]['y'] * -1.0,
                 camera_df.iloc[index]['z'] * -1.0]

    # Camera's intrinsic matrix
    camera_im = [[f_x, 0.0, c_x],
                 [0.0, f_y, c_y],
                 [0.0, 0.0, 1.0]]
    camera_im = np.float32(camera_im)

    # Distortion coefficients matrix
    cof_m = [camera_df.iloc[index]['coe3'], camera_df.iloc[index]['coe4'], camera_df.iloc[index]['coe5'], 0.0, 0.0]
    cof_m = np.float32(cof_m)

    # Converting 1x4 orientation (rotation) matrix to 3x3
    divd = math.sqrt(1 - camera_df.iloc[index]['o4'] * camera_df.iloc[index]['o4'])
    om_1 = [[camera_df.iloc[index]['o1'] / divd],
            [camera_df.iloc[index]['o2'] / divd],
            [camera_df.iloc[index]['o3'] / divd]]
    om_1 = np.float32(om_1)
    om_converted, _ = cv2.Rodrigues(om_1)

    # 3x4 Camera Matrix
    cm_1 = [[om_converted[0][0], om_converted[0][1], om_converted[0][2], t_mtrx[0]],
            [om_converted[1][0], om_converted[1][1], om_converted[1][2], t_mtrx[1]],
            [om_converted[2][0], om_converted[2][1], om_converted[2][2], t_mtrx[2]]]
    cm_1 = np.matmul(camera_im, cm_1)
    cm_1 = np.float32(cm_1)
    camera_1 = camera.Camera(index, camera_df.iloc[index]['did'], f, c_x, c_y, f_x, f_y,
                              camera_im, t_mtrx, om_converted, cof_m, rel_pos_1, cm_1)
    return camera_1

def choose(X,n=1):
    return np.random.choice(X,n)

def xyz_in_volume(xyz, volume):
    x,y,z = xyz
    x_inrange = np.logical_and(volume[0][0]<=x, volume[0][1]>=x)
    y_inrange = np.logical_and(volume[1][0]<=y, volume[1][1]>=y)
    z_inrange = np.logical_and(volume[2][0]<=z, volume[2][1]>=z)

    if np.all([x_inrange, y_inrange, z_inrange]):
        return True
    else:
        return False

def make_brownian_particles(n_points, xyz_volume, frames=25, stepsize=0.05):
    '''
    Makes #d trajectories for N particles showing Brownian type motion. 
    All motion is constrained to happen within a 3D bounding cuboid
    set by xyz_volume. 

    Parameters
    ----------
    n_points : int>0
    xyz_volume : list with 3 sub-lists
        With structure [[xmin,xmax], [ymin, ymax], [zmin, zmax]]
    frames : int>0, optional 
        Number of frames to generate motion for. Defaults to 25
    stepsize : float>0, optional 
        Step size defines the standard deviation of the x,y,z jumps for the 
        particles. Essentially x[t+1] = x[t] + normal(0,stepsize) and same for y and z

    Returns 
    -------
    all_particle_paths : pd.DataFrame 
        With columns: frame, oid, x, y, z
    '''
    xmin, xmax = xyz_volume[0]
    ymin, ymax = xyz_volume[1]
    zmin, zmax = xyz_volume[2]

    xrange = np.arange(xmin, xmax+stepsize, stepsize)
    yrange = np.arange(ymin, ymax+stepsize, stepsize)
    zrange = np.arange(zmin, zmax+stepsize, stepsize)

    particle_paths = {}
    for particle in range(n_points):
        particle_paths[particle] = []
        xyz_start = np.array([choose(axis) for axis in [xrange, yrange, zrange]]).flatten()
        particle_paths[particle].append(xyz_start)
        for frame in range(frames-1):
            notinframe = True
            if frame==0: 
                while notinframe:
                    candidate_xyz = xyz_start + np.random.normal(0,stepsize,3)
                    notinframe = not xyz_in_volume(candidate_xyz, xyz_volume)    
                particle_paths[particle].append(candidate_xyz)
            else:
                while notinframe:
                    jump = np.random.normal(0, stepsize, 3)
                    trial_candidate_xyz = candidate_xyz + jump
                    notinframe = not xyz_in_volume(trial_candidate_xyz, xyz_volume)
                candidate_xyz = trial_candidate_xyz.copy()
                particle_paths[particle].append(candidate_xyz)
        particle_paths[particle] = np.array(particle_paths[particle]).reshape(-1,3)

    trajectories = []
    for particle, trajectory in particle_paths.items():
        df = pd.DataFrame(data={'frame': np.arange(frames),
                          'x': trajectory[:,0],
                          'y':trajectory[:,1],
                          'z': trajectory[:,2],
                          'oid':particle})
        trajectories.append(df)
    all_particle_paths = pd.concat(trajectories).reset_index(drop=True)
    return all_particle_paths
                
                    

def find_best_pair_of_cameras(camera_df, gt_3d_df):
    '''
    Projects 3D groundtruth data onto each camera, and checks the number
    of points with valid 2D projections. 
    
    NOT TESTED -- NEEDS TO BE CHECKED...
    
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

def make_rotation_mat_fromworld(Rc, C):
    '''
    Parameters
    ----------
    Rc : 3x3 np.array
        Rotation matrix wrt the world coordinate system
    C : (1,3) or (3,) np.array
        Camera XYZ in world coordinate system

    Returns
    -------
    camera_rotation: 4x4 np.array
        The final camera rotation and translation matrix which 
        converts the world point to the camera point

    
    References
    ----------
    * Simek, Kyle, https://ksimek.github.io/2012/08/22/extrinsic/ (blog post) accessed 14/2/2022

    
    Example
    -------
    
    > import track2trajectory.synthetic_data as syndata
    > Rc, C = ..... # load and define the world system camera rotations and camera centre
    > rotation_mat = make_rotation_mat_fromworld(Rc, C)

    '''
    camera_rotation = np.zeros((4,4))
    camera_rotation[:3,:3] = Rc.T
    camera_rotation[:3,-1] = -np.matmul(Rc.T,C)
    camera_rotation[-1,-1] = 1 
    return camera_rotation

def make_focal_mat(focal_m):
    '''

    Parameters
    ----------
    focal_m : float
        Focal length of the camera in metres

    Returns
    -------
    focal_mat : 3x4 np.array

    '''
    focal_mat = np.zeros((3,4))
    focal_mat[0,0] = focal_m
    focal_mat[1,1] = focal_m
    focal_mat[2,2] = 1 
    return focal_mat

def get_cam_centre_from_projectionmat(P):
    '''
    
    ::
        C = -M^{-1} x p4
    
    Where M is the top-left 3x3 matrix of the 4x4 Projection matrix
    and p4 is the 4th column of the Projection matrix.

    Parameters
    ----------
    P :(4,4) np.array
        camera projection matrix
    Returns 
    -------
    cam_C : (1,3) np.array
        Camera centre in world coordinates
    '''
    M = P[:3,:3]
    p4 = P[:3,-1]
    # in world coordinates
    cam_C = np.matmul(-np.linalg.inv(M),p4)
    return cam_C


def get_cam_coods(focal_m, R_mat, world_point):
    '''
    Get 2D points from a 3D world point using the 
    intrinsic and rotation matrices
    
    Parameters
    ----------
    focal_m : float >0 
        focal length in m
    R_mat : 4x4 np.array 
        R matrix with 3x3 rotation matrix and translation column
    world_point : (3,) or (1,3) np.array
        X,Y,Z coordinates 

    Returns
    -------
    cam_point : (3,) np.array
        x,y,z coordinates in camera coordinate system
    '''
    # in camera global coords where Z is pointing out
    rearranged_xzy = np.array([world_point[0], world_point[2], world_point[1]])
    worldpoint_homog = np.concatenate((rearranged_xzy, [1]))
    focal_mat = make_focal_mat(focal_m)
    focal_Rmat = np.matmul(focal_mat, R_mat)
    cam_point = np.matmul(focal_Rmat, worldpoint_homog)
    return cam_point
