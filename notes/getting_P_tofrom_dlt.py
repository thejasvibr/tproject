# -*- coding: utf-8 -*-
"""
Getting P from DLT coefficients
===============================


References
----------
*DLT to/from intrinsic+extrinsic : https://biomech.web.unc.edu/dlt-to-from-intrinsic-extrinsic/
accessed 2022-02-27
"""
import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance, distance_matrix
from scipy.spatial.transform import Rotation
from track2trajectory.synthetic_data import get_cam_centre_from_projectionmat
#%% The 12 DLT coefficients are part of a 3x4 matrix, where the last entry 
# DLT[4,4] is a scaling factor. The coefficients are laid out so
#::
#    [ co1, co2, co3,   co4]
#    [ co5, co6, co7,   co8]
#    [ co9, co10, c011, co12]
#
# The DLT matrix can be obtained by the relation: :math:`DLT = K \times m \times P`
# where `K` is the intrinsic matrix
# ::
#   [fx 0 px]
#   [0 fy py]
#   [0 0   1] 
#
# Where fx and fy are the horizontal & vertical focal lengths and px,py are the 
# hor/vert image centres
# `m` is a 'transition' matrix to make the matrix shapes compatible. 
# ::
#    [1 0 0  0]
#    [0 1 0  0]
#    [0 0 1  0]
#    [0 0 0  0]


#%% Let's first simulate certain situations 
fx, fy = [1230]*2
px, py = 960, 540
K = np.array([[fx, 0, px ],
              [0, fy, py],
              [0,  0,  1]])

#%% Make the `m` matrix
m = np.zeros((3,4))
for (row,col) in ([0,0],[1,1],[2,2]):
    m[row,col] = 1

#%% The projection matrix has the rotation and translation matrices as sub-matrices
# Let's first create the simplest possible situation, where the camera is at the 
# origin and translated by 10 units in x and y Cworld = [10, 10, 0]
Cworld = np.array([10, 10, 1])
T = np.concatenate((-Cworld, np.array([1])))
R = Rotation.from_euler('xyz', [5,10,5],degrees=True).as_matrix()


P = np.zeros((4,4))
P[:,-1] = T
P[:3,:3] = R

#%% get the DLT coefficients
dlt = np.matmul(np.matmul(K,m),P)

#%% Now from the dlt, let's try to get the P matrix back.
Km = np.matmul(K,m)

Psim_est, residual, _ ,_ = np.linalg.lstsq(Km, dlt)

#%% Now let's try the same trick on the TeAx cameras from the cave. 
# We have the DLT coefficients from the 
dltcoefs = pd.read_csv('2018-08-17_wand_dltCoefs.csv',header=None)
c1_dlt, c2_dlt, c3_dlt = [dltcoefs.loc[:,col] for col in [0,1,2]]
# camera image has 640 x 512
px,py = 320, 256
fx = 526 # in pixels

Kteax = np.array([[fx, 0, px],
                  [0, fy, py],
                  [0, 0,  1]])
#%% 
def assign_coefs_rowwise(coefs):
    dlt_mat = np.zeros((3,4))
    k = 0 
    for i in range(dlt_mat.shape[0]):
        for j in range(dlt_mat.shape[1]):
            if not k>10:
                dlt_mat[i,j] = coefs[k]
                k+=1
            else:
                pass
    return dlt_mat
# Cam1
dlt_teax_c1 = assign_coefs_rowwise(c1_dlt)
Km_teax = np.matmul(Kteax, m)
Pest_cam1, res, _, _= np.linalg.lstsq(Km_teax, dlt_teax_c1)
R_cam1 = Pest_cam1[:3,:3]

# Cam2 
dlt_teax_c2 = assign_coefs_rowwise(c2_dlt)
Pest_cam2, res, _, _= np.linalg.lstsq(Km_teax, dlt_teax_c2)
R_cam2 = Pest_cam2[:3,:3]

# Cam 3
 
dlt_teax_c3 = assign_coefs_rowwise(c3_dlt)
Pest_cam3, res, _, _= np.linalg.lstsq(Km_teax, dlt_teax_c3)
R_cam3 = Pest_cam3[:3,:3]


#%% 
cam1_centre = get_cam_centre_from_projectionmat(Pest_cam1)
cam2_centre = get_cam_centre_from_projectionmat(Pest_cam2)
cam3_centre = get_cam_centre_from_projectionmat(Pest_cam3)
camera_centres = [cam1_centre, cam2_centre, cam3_centre]

cam2cam_distances = distance_matrix(camera_centres,
                                  camera_centres)

plt.figure(figsize=plt.figaspect(1)*2)
ax = plt.subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])
for each in [cam1_centre, cam2_centre, cam3_centre]:
    ax.plot(each[0],each[1],each[2],'*')
