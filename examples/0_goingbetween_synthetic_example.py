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

"""

import cv2
import matplotlib.pyplot as plt
import numpy as np 
np.set_printoptions(precision=8)
np.set_printoptions(suppress=True)
import track2trajectory.synthetic_data as syndata
from track2trajectory.projection import project_to_2d_and_3d
from dltrecon import DLTrecon
from track2trajectory.dlt_to_world import transformation_matrix_from_dlt
from scipy.spatial import distance
from scipy.spatial.transform import Rotation
from dlt_reconstruction_easyWand import dlt_reconstruct
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
cam1_R = Rotation.from_euler('xyz', [5,-10,0],degrees=True).as_matrix()
cam2_R = Rotation.from_euler('xyz', [2,10,0],degrees=True).as_matrix()
cam1_C = [-1,0,1]
cam2_C = [1,0,-0.25]
cam1, cam2 = syndata.generate_two_synthetic_cameras_version2(cam2_Rotation=cam2_R,
                                                             cam1_Rotation=cam1_R,
                                                             cam1_C=cam1_C,
                                                             cam2_C=cam2_C)
objects_in_3d = syndata.make_brownian_particles(3,[[-2,2],[6,8],[-1,1]], frames=2)
cam1_pixels, _ = project_to_2d_and_3d(objects_in_3d, cam1)
cam2_pixels, _ = project_to_2d_and_3d(objects_in_3d, cam2)

cam1_xy = cam1_pixels.loc[:,['x','y']].to_numpy()
cam2_xy = cam2_pixels.loc[:,['x','y']].to_numpy()
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

revP_distmat = point2point_matrix(rev_Pbased)
print('Ground truth - DLT based distance matrix', pbased_distmat-revP_distmat)

#%%
# This examples with synthetic data shows two things: 1) it is indeed possible to 
# back and forth between the projection matrix and DLT formulations and get the `same`
# xyz coordinates 2) the established workflows in my codebase are correct. 
# 
# Of course, one major point of difference here is that my synthetic data has no 
# noise - be it in the 2D pixel coordinates, or in the intrinsic/extrinsic parameter
# estimation, and/or DLT coefficient estimation. Perhaps the noiseless nature of
# the 



