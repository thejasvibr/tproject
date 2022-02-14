# -*- coding: utf-8 -*-
"""
Notes of Camera Matrices
========================
P = diag(f,f,1)[I|0] is the short for 

P = [f 0 0 0]
    [0 f 0 0]
    [0 0 1 0]


References
----------
* Hartley & Zisserman, Multiple View Geometry, Chapter 6
* https://ksimek.github.io/2012/08/22/extrinsic/ (blog post) access 14/2/2022
"""
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np 
from scipy.spatial.transform import Rotation
import track2trajectory.synthetic_data as syndata

def get_cam_centre_and_direction(rotmat, transmat):
    cam_centre = np.matmul(rotmat.T, transmat)
    cam_direction = -np.matmul(rotmat.T, transmat)
    return cam_centre, cam_direction

# thanks to https://stackoverflow.com/a/51512965/4955732
normalized_v = lambda v: v / np.sqrt(np.sum(v**2))


cam1_R = np.eye(3) # aligned to world coordinates - no rotations in x,y,z
cam1_T = np.zeros(3) 


cam2_R = Rotation.from_euler('y', 0, 'degrees').as_matrix()
# cam2_R = np.array(  [[0.98480775301, 0.0, 0.17364817766],
#                     [0.0,           1.0, 0.0],
#                     [0.17364817766 * - 1.0, 0.0, 0.98480775301]])
cam2_T =  np.array([-2.0, 0.0, 0.0])


cam3_R = Rotation.from_euler('z', 45, 'degrees').as_matrix()
cam3_T = np.array([-3, 0.0, 0.0])


cam1_C, cam1_dir = get_cam_centre_and_direction(cam1_R, cam1_T)
cam2_C, cam2_dir = get_cam_centre_and_direction(cam2_R, cam2_T)
cam3_C, cam3_dir = get_cam_centre_and_direction(cam3_R, cam3_T)

#%%
# this direction vector is in the camera coordinate system. 
# let's get the vector in the world coordinate system. 
cam2_mat = np.zeros((4,4))
cam2_mat[:3,3] = cam2_T
cam2_mat[:3,:3] = cam2_R 
cam2_mat[3,3] = 1

focal_mat = np.zeros((3,4))
focal = 0.015 # m
focal_mat[0,0] = focal
focal_mat[1,1] = focal
focal_mat[2,2] = 1

world_point = np.array([2,2,2,1])

f_cammat = np.matmul(focal_mat,cam2_mat)
cam_coords = np.matmul(f_cammat, world_point)
print(cam_coords)
print(cam_coords/cam_coords[-1])

# Pc = R Pw , here R is the cam2_mat
# Pw = R-1 Pc
cam2_dir_homog = np.concatenate((cam2_dir, [1]))
cam2_dir_world = np.matmul(cam2_mat.T, cam2_dir_homog)


#%% 
mat1 =  np.row_stack(([0,0,0],
                        cam1_dir))
mat2 = np.row_stack((cam2_C,
                     cam2_C+normalized_v(cam2_dir)))
mat3 = np.row_stack((cam3_C,
                     cam3_C+normalized_v(cam3_dir)))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(mat1[0,0], mat1[0,1], mat1[0,2], '*')
ax.plot3D(mat2[0,0], mat2[0,1], mat2[0,2], '*')
ax.plot3D(mat3[0,0], mat3[0,1], mat3[0,2], '*')

ax.plot3D(mat1[:,0], mat1[:,1], mat1[:,2], )
ax.plot3D(mat2[:,0], mat2[:,1], mat2[:,2], )
ax.plot3D(mat3[:,0], mat3[:,1], mat3[:,2], )


ax.set_ylim(-11,11)
ax.set_zlim(-12,12)
ax.set_xlim(-11,11)


#%% 
# Converting Rotation matrix to Euler angles
# code from https://learnopencv.com/rotation-matrix-to-euler-angles/
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])
#%%
# make the 3 cameras
c1,c2,c3 = syndata.generate_three_synthetic_cameras()
c1.r_mtrx

euler_angles = np.degrees(rotationMatrixToEulerAngles(c3.r_mtrx))
print(euler_angles)

get_cam_centre_and_direction(c1.r_mtrx, c1.t_mtrx)
