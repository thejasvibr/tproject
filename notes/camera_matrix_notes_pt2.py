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



References
----------
* https://ksimek.github.io/2012/08/22/extrinsic/ (blog post) accessed 14/2/2022
"""
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np 
from scipy.spatial.transform import Rotation


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
    focal_mat[0,1] = ppx
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

Rc_cam1 = Rotation.from_euler('xyz',[0,0,0],degrees=True).as_matrix()
C_cam1 = np.zeros(3)
cam1_rotmat = make_rotation_mat_fromworld(Rc_cam1, C_cam1)

#%% and now let's project a point in the world to the camera 
fmm = 0.005
f_pixels = 1230 # pixels
world_point = np.array([4,0.25,0.25]) # xyz in 'standard' cartesian

camccods = get_cam_coods(1230, 960, 540, cam1_rotmat, world_point)
print(camccods/camccods[-1])




