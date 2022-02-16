# -*- coding: utf-8 -*-
"""
Notes of Camera Matrices aprt 2
===============================
The 'standard' way to represent the camera matrix is the world to camera mapping
which is not intuitive at all. 

However, we typically understand the camaera in the world mapping better.Camera
in the world coordinate system is better imagined intuitively in terms of camera
pose and the camera centre's location.

The extrinsic matrix is thus

| Rc.T | -Rc.T C|
|---- |------ |
|  0  |   1   |

Where Rc is the camera pose wrt world coordinates and C is the camera centre
in world XYZ.

XYZ vs 'XZY'
------------
I'm having the trouble of wrapping my head around the camera and world coordinate 
systems. While the world coordinate system has the 'normal' XYZ order of axes, where
z is up/down and y is front/back., it seems like the camera coordinate system 
has z: front/back, y:up/down and the order is also always xyz too.

I need to maintain this order even for the rotation matrices it seems like!


References
----------
* https://ksimek.github.io/2012/08/22/extrinsic/ (blog post) accessed 14/2/2022
"""
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np 
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

def make_rotation_mat_fromworld(Rc, C):
    '''
    Given the camera Centre and Rotation matrix in World coordinates, 
    generates the world->camera R matrix.
    
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
    * https://ksimek.github.io/2012/08/22/extrinsic/ (blog post) accessed 14/2/2022
    '''
    camera_rotation = np.zeros((4,4))
    camera_rotation[:3,:3] = Rc.T
    camera_rotation[:3,-1] = -np.matmul(Rc.T,C)
    camera_rotation[-1,-1] = 1 
    return camera_rotation

def make_focal_mat(focal_m, ppx, ppy):
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
    focal_mat[0,2] = ppx
    focal_mat[1,2] = ppy
    focal_mat[1,1] = focal_m
    focal_mat[2,2] = 1 
    return focal_mat

def get_cam_coods(focal_m, ppx, ppy, R_mat, world_point):
    '''
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
    rearranged_xzy = np.array([world_point[0], world_point[1], world_point[2]])
    worldpoint_homog = np.concatenate((rearranged_xzy, [1]))
    focal_mat = make_focal_mat(focal_m, ppx, ppy)
    focal_Rmat = np.matmul(focal_mat, R_mat)
    cam_point = np.matmul(focal_Rmat, worldpoint_homog)
    return cam_point

# let's create a camera aligned with world xyz and located at 0,0,0
f_pixels = 1230 # pixels
ppx, ppy = 960, 540

world_point = np.array([0,10,0], dtype='float32') # xyz in 'standard' cartesian
world_point_camerastyle = np.float32([world_point[0], world_point[2], world_point[1]]) # in xzy style

# always remember Z points in the front/back direction!!!
Rc_cam1 = Rotation.from_euler('xyz',[10,10,0],degrees=True).as_matrix()
C_cam1 = np.zeros(3)
cam1_rotmat = make_rotation_mat_fromworld(Rc_cam1, C_cam1)


r_matrix = np.float32(cam1_rotmat[:3,:3])
t_matrix = np.float32(cam1_rotmat[:3,-1])
focal_mat = np.float32(make_focal_mat(f_pixels, ppx, ppy))[:,:-1]


camccods = get_cam_coods(1230, 960, 540, cam1_rotmat, world_point_camerastyle)
normalised_coods = camccods/camccods[-1]
print(normalised_coods[:-1], camccods)


rvec, _ = cv2.Rodrigues(r_matrix)
dist_coefs = np.float32(np.zeros(4))
twod_point, _ = cv2.projectPoints(world_point_camerastyle, rvec,
                                          t_matrix, focal_mat, dist_coefs)
print(f'opencv {twod_point}')

