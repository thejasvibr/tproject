# -*- coding: utf-8 -*-
"""
Going between DLT and intrinsic/extrinsic formulations
======================================================
I've had issues going between two common camera triangulation 
formulations (what I call the Projection matrix method, and the commonly known DLT method) 
when using experimental data. Experimental data is of course bound to be noisy, and
so here I'll use synthetic data with no noise to solidify my understanding.

I'll first generate 2 cameras and a few moving objects. Then project the 
objects onto the camera planes, and finally try to recover the 3D coordinates.


Some background
~~~~~~~~~~~~~~~
The simulated data is created using the codebase developed by Giray Tandogan and 
modified by Thejasvi Beleyur. It creates a group of objects moving randomly within a 
3D volume and such that the points are visible to both cameras on all frames.

Both cameras are `not` in plane, don't have any distortion - and have sensor and
focal length that match a Go Pro Hero (exact model here). 

References
~~~~~~~~~~
* DLT to/from intrinsic+extrinsic https://biomech.web.unc.edu/dlt-to-from-intrinsic-extrinsic/ acc 2022-03-08
* http://www.kwon3d.com/theory/dlt/dlt.html#3d



Author: Thejasvi Beleyur, March 2022
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
np.set_printoptions(precision=8)
np.set_printoptions(suppress=True)
import track2trajectory.synthetic_data as syndata
from track2trajectory.projection import project_to_2d_and_3d
from track2trajectory.dlt_to_world import transformation_matrix_from_dlt
from track2trajectory.dlt_to_world import cam_centre_from_dlt
from scipy.spatial import distance
from scipy.spatial.transform import Rotation
from dlt_reconstruction_easyWand import dlt_reconstruct
from track2trajectory.synthetic_data import make_rotation_mat_fromworld

np.random.seed(82319)

def point2point_matrix(xyz):
    distmat = np.zeros((xyz.shape[0],xyz.shape[0]))
    for i in range(distmat.shape[0]):
        for j in range(distmat.shape[1]):
            distmat[i,j] = distance.euclidean(xyz[i,:], xyz[j,:])
    return distmat
#%%
# Generate the synthetic data
# Let's also rotate the second camera a bit to the origin
# The rotations are around the `camera` axis wrt World axis
# i.e. x,y is on image plane and z points out into the 'depth'
# Both cameras aren't really aligned completely with the 'world' axis.

cam1_R = Rotation.from_euler('xyz', [5,-10,2],degrees=True).as_matrix() 
cam2_R = Rotation.from_euler('xyz', [1,10,-4],degrees=True).as_matrix()
cam1_C = [-1,0,1] # camera centres in world XYZ, Y faces into the depth, Z goes up/down
cam2_C = [1,0.5,-0.25]
# generate camera instances with pre-specified parameters
cam1, cam2 = syndata.generate_two_synthetic_cameras_version2(cam2_Rotation=cam2_R,
                                                             cam1_Rotation=cam1_R,
                                                             cam1_C=cam1_C,
                                                             cam2_C=cam2_C)
# generate brownian type particles that move within a specified bounding box
objects_in_3d = syndata.make_brownian_particles(3,[[-2,2],[6,8],[-1,1]], frames=2)
# project 3D particle position to 2D camera pixel positions
cam1_pixels, _ = project_to_2d_and_3d(objects_in_3d, cam1)
cam2_pixels, _ = project_to_2d_and_3d(objects_in_3d, cam2)

cam1_xy = cam1_pixels.loc[:,['x','y']].to_numpy()
cam2_xy = cam2_pixels.loc[:,['x','y']].to_numpy()

print('Ground truth points')
objects_in_3d

#%% 
# Verify Projection method: it works
# ----------------------------------
# Now using the projection matrix let's get the 3D positions back using the 
# intrinsic/extrinsic methods based on the 3x4 projection matrix P. 

threed_Pbased = []
for (xy1, xy2) in zip(cam1_xy, cam2_xy):
    pos_homog = cv2.triangulatePoints(cam1.cm_mtrx, cam2.cm_mtrx, xy1, xy2)
    xyz_position = cv2.convertPointsFromHomogeneous(pos_homog.T)
    threed_Pbased.append(xyz_position)
threed_Pbased = np.array(threed_Pbased).reshape(-1,3)
pbased_distmat = point2point_matrix(threed_Pbased)

#%% 
# We see the y-z axes are interchanged, but consistently so. 
print('Projection matrix based triangulation:')

threed_Pbased

#%%
# DLT coefficients from Projection matrix: it works
# -------------------------------------------------
# Let's go from the projection matrix to DLT coefficients and try reconstructing
# the 3D positions using the DLT method. 
fx, fy  = [1230,1230]
px, py = 960, 540
K = np.array([[fx, 0, px],
              [0, fy, py],
              [0, 0,  1]])
m = np.column_stack((np.eye(3), np.zeros(3)))

#%% 
# Now, follow the 12 entry -> 11 entry conversion by normalising the 
# first 11 entries with the 12th entry. The first 11 entries then become the 
# 11 DLT coefficients used to triangulate.
dlt_c1 = cam1.cm_mtrx.flatten()[:-1]/cam1.cm_mtrx[-1,-1]
dlt_c2 = cam2.cm_mtrx.flatten()[:-1]/cam2.cm_mtrx[-1,-1]
dlt_11coefs_fromP = np.column_stack((dlt_c1, dlt_c2))

xyz_11dlt_P = []
for (xy1, xy2) in zip(cam1_xy, cam2_xy):
    object_pixels = np.append(xy1, xy2).reshape(1,-1)
    xyz = dlt_reconstruct(dlt_11coefs_fromP, object_pixels)[0]
    xyz_11dlt_P.append(xyz)
xyz_11dlt_fromP = np.array(xyz_11dlt_P).reshape(-1,3)

pd.DataFrame(dlt_11coefs_fromP, columns=['cam1DLT','cam2DLT'])


#%% 
# Aside from the y-z axes interchange here too - the values match up. 

print('DLT from projection matrix triangulation:')

pd.DataFrame(xyz_11dlt_fromP, columns=['x','y','z'])


#%% 
# Projection matrix from DLT 
# --------------------------
# Okay, so now let's do the `reverse`. Let's say we have the DLT coefficients. 
# How do we go about getting the `projection matrix`?
# In the Hedrick write-up, `P` is actually not the Projection matrix as 
# typically understood (which maps 2d<->3d, `x = P X`, where P = K[R|t], where R is
# rotation and t is the translation). Instead it seems that P = [R|t] a 4x4 matrix which contains
# the rotation and translation info.

# To fill the 3x4 matrix with 12 entries from the (normalised) 11 DLT coefficients,
# let's append a 1 at the end.

rev_P_cam1 = np.append(dlt_11coefs_fromP[:,0], [1]).reshape(3,4)
rev_P_cam2 = np.append(dlt_11coefs_fromP[:,1], [1]).reshape(3,4)

rev_Pbased = []
for (xy1, xy2) in zip(cam1_xy, cam2_xy):
    pos_homog = cv2.triangulatePoints(rev_P_cam1, rev_P_cam2, xy1, xy2)
    xyz_position = cv2.convertPointsFromHomogeneous(pos_homog.T)
    rev_Pbased.append(xyz_position)
rev_Pbased = np.array(rev_Pbased).reshape(-1,3)

print('Projection matrix from DLT:')
pd.DataFrame(rev_Pbased, columns=['x','y','z'])

#%%
# This examples with synthetic data shows two things: 1) it is indeed possible to 
# back and forth between the projection matrix and DLT formulations and get the `same`
# xyz coordinates 2) the established workflows in my codebase are correct. 
# 
# Of course, one major point of difference here is that my synthetic data has no 
# noise - be it in the 2D pixel coordinates, or in the intrinsic/extrinsic parameter
# estimation, and/or DLT coefficient estimation. Perhaps the noiseless nature of
# the 
#
# Here, we've shown that we can recover the ground truth 3D coordinates
# of a synthetic data set using multiple methods 1) 'standard' projection matrix
# based methods 2) obtaining 11 DLT coefficients from the 3x4 projection matrix
# and finally 3) 'regenerating' the 3x4 projection matrix from the 11 DLT coefficients
# at hand. The coordinates match the ground truth extremely well (<1e-6)



#%% 
# Getting the projection matrix from `DLTcameraPosition`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The function :code:`transformation_matrix_from_dlt` is my Python implementation of 
# Ty Hedrick's `DLTcameraPosition`. :code:`transformation_matrix_from_dlt` is correct
# as it calculates the expected principal points, and focal lengths of the cameras. 
# The T1, T2 matrices are also numerically equivalent to the function output in Octave.

T1, z1, ypr1 = transformation_matrix_from_dlt(dlt_11coefs_fromP[:,0]) # python implementation of DLTcameraPosition
T2, z2, ypr2 = transformation_matrix_from_dlt(dlt_11coefs_fromP[:,1])

#T1 = T1.T; T2= T2.T # transpose and convert the T matrix into the 'classic'
# left-upper 3x3 rotation mat and the last column with the translation

print('DLTcameraPosition T matrices: \n \n ', T1, '\n \n ' ,T2)

# shifts the left/right-handed to the other (it's still magic as of now)
shifter_mat = np.row_stack(([1,0,0,0],
                            [0,1,0,0],
                            [0,0,-1,0],
                            [0,0,0,1]))
shifted_rotmat1 = np.matmul(T1, shifter_mat)[:3,:3]
shifted_rotmat2 = np.matmul(T2, shifter_mat)[:3,:3]

#%% 
# Convert world coordiantes to camera frame to generate the projection matrix
T1cam = make_rotation_mat_fromworld(shifted_rotmat1, T1[-1,:3])
T2cam = make_rotation_mat_fromworld(shifted_rotmat2, T2[-1,:3])

print('T matrices converted to camera frame \n \n', T1cam, ' \n \n ' ,T2cam)

fx, fy = cam1.f_x, cam1.f_y

K = np.array(([fx, 0, cam1.c_x],
              [0, fy, cam1.c_y],
              [0, 0,     1    ]))
Km = np.matmul(K,m)
m = np.column_stack((np.eye(3), np.zeros(3))) # the 3x4 matrix to 'grease the wheels'

P11kmt = np.matmul(Km, T1cam)
P22kmt = np.matmul(Km, T2cam)

xyz_tmat_based = []
for (xy1, xy2) in zip(cam1_xy, cam2_xy):
    pos_homog = cv2.triangulatePoints(P11kmt, P22kmt, xy1, xy2)
    xyz_position = cv2.convertPointsFromHomogeneous(pos_homog.T)
    xyz_tmat_based.append(xyz_position)

xyz_tmat_based = np.array(xyz_tmat_based).reshape(-1,3)

print('XYZ coordinates generated through DLTcameraPosition inputs: \n ')
pd.DataFrame(xyz_tmat_based, columns=['x','y','z'])
