"""
Matching and running on experimental data
=========================================
In the previous example (`Quick example`) we saw how to 2D matching ran on a simulated
dataset. Here we will illustrate the expected workflow when users actually try 
to use their own datasets with all the 'real-world' details in them. 


@author: theja
"""
import glob
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np 
from track2trajectory.camera import Camera
from track2trajectory.match3d import match_2dpoints_to_3dtrajectories
from track2trajectory.projection import calcFundamentalMatrix
import dltrecon as dltr
from dlt_reconstruction_easyWand import dlt_reconstruct
from track2trajectory.synthetic_data import make_rotation_mat_fromworld, get_cam_centre_from_projectionmat
from track2trajectory.dlt_to_world import transformation_matrix_from_dlt, cam_centre_from_dlt
import mat73
from scipy.spatial import distance
from gravity_alignment import smooth_and_acc, row_calc_norm

concat = np.concatenate

def get_point2point_matrix(xyz):
    distmat = np.zeros((xyz.shape[0],xyz.shape[0]))
    for i in range(distmat.shape[0]):
        for j in range(distmat.shape[1]):
            distmat[i,j] = distance.euclidean(xyz[i,:], xyz[j,:])
    return distmat

#%% Undistorting pixel coordinates
#   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Most, if not all, cameras have some form of distortion. Once we know the distortion
# coefficients, it is easy to 'undistort' an image or an object's pixel coordinates,
# and thus generate corrected object coordinates. 

# load the raw 2D tracking data
multicam_xy = pd.read_csv('DLTdv7_data_gravityxypts.csv')
colnames = multicam_xy.columns
cam1_2d, cam2_2d, cam3_2d = [multicam_xy.loc[:,colnames[each]].to_numpy(dtype='float32') for each in [[0,1], [2,3], [4,5]]]

# the pixel data was digitised using DLTdv (https://biomech.web.unc.edu/dltdv/)
# The origin in DLTdv is set to the lower left, and so the y-axis needs
# to be flipped. 

cam1_2d[:,1] = 511 - cam1_2d[:,1]
cam2_2d[:,1] = 511 - cam2_2d[:,1]
cam3_2d[:,1] = 511 - cam3_2d[:,1]


#%% Load the dlt coefficients - and infer the Projection matrix. We already
# know the intrinsic matrix (common to all cameras)

# camera image has 640 x 512
px,py = 320, 256
fx, fy = 526, 526 # in pixels

Kteax = np.array([[fx, 0, px],
                  [0, fy, py],
                  [0, 0,  1]])

#%% All cameras are assumed to have the same distortion, and so we'll apply the 
# same undistortion to all of them. 
p1, p2 = np.float32([0,0]) # tangential distortion
k1, k2, k3 = np.float32([-0.3069, 0.1134, 0]) # radial distortion
dist_coefs = np.array([k1, k2, p1, p2, k3]) #in the opencv format

# apply undistortion now
cam1_undist = cv2.undistortPoints(cam1_2d, Kteax, dist_coefs, P=Kteax)
cam2_undist = cv2.undistortPoints(cam2_2d, Kteax, dist_coefs, P=Kteax)
cam3_undist = cv2.undistortPoints(cam3_2d, Kteax, dist_coefs, P=Kteax)

#%%
plt.figure()
plt.subplot(311)
plt.plot(cam1_2d[:,0], cam1_2d[:,1],'*')
plt.plot(cam1_undist[:,:,0], cam1_undist[:,:,1],'^')
plt.ylim(512,0)
plt.xlim(0,640)

plt.subplot(312)
plt.plot(cam2_2d[:,0], cam2_2d[:,1],'*')
plt.plot(cam2_undist[:,:,0], cam2_undist[:,:,1],'^')
plt.ylim(512,0)
plt.xlim(0,640)

plt.subplot(313)
plt.plot(cam3_2d[:,0], cam3_2d[:,1],'*')
plt.plot(cam3_undist[:,:,0], cam3_undist[:,:,1],'^')
plt.ylim(512,0)
plt.xlim(0,640)

#%%
# Now reformat into the dataframe format required:
import uuid
cam1df = pd.DataFrame(data={'x':cam1_undist[:,:,0].flatten(),
                            'y': cam1_undist[:,:,1].flatten(),
                            'oid':[str(uuid.uuid4())[-4:] for each in range(cam1_undist.shape[0])],}
                      )

cam2df = pd.DataFrame(data={'x':cam2_undist[:,:,0].flatten(),
                            'y': cam2_undist[:,:,1].flatten(),
                            'oid':[str(uuid.uuid4())[-4:] for each in range(cam2_undist.shape[0])]}
                      )

cam3df = pd.DataFrame(data={'x':cam3_undist[:,:,0].flatten(),
                            'y': cam3_undist[:,:,1].flatten(),
                            'oid':[str(uuid.uuid4())[-4:] for each in range(cam3_undist.shape[0])]}
                      )

for each in [cam1df, cam2df, cam3df]:
    each['frame'] = np.arange(each.shape[0])

#%%
# DLT reconstruction using 11 parameter DLT from easyWand
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fname = '2018-08-17_wand_dvProject.mat'

data_dict = mat73.loadmat(fname)
dltcoefs = data_dict['udExport']['data']['dltcoef']

c1_dlt, c2_dlt, c3_dlt = [dltcoefs[:,col] for col in [0,1,2]]


coefficents = np.column_stack((c1_dlt, c2_dlt))
xyz_easywand = []
for i in range(11,24):
    points = np.append(np.float32(cam1df.loc[i,['x','y']]), np.float32(cam2df.loc[i,['x','y']])).reshape(1,-1)
    xyz_easywanddlt = dlt_reconstruct(coefficents, points)
    xyz_easywand.append(xyz_easywanddlt[0])
xyz_dlt_easywand = np.array(xyz_easywand).reshape(-1,3)

#%%
# DLT -> Transformation-> Projection matrix
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# First we get the transformation matrix from the DLT coefficients
# and then following the `Ty Hedrick and Taila Weiss write-ups here <https://biomech.web.unc.edu/dlt-to-from-intrinsic-extrinsic/>`_
# we'll get the final projection matrix :math:`P`.
#
#

T1 = transformation_matrix_from_dlt(dltcoefs[:,0])
T2 = transformation_matrix_from_dlt(dltcoefs[:,1])
T3 = transformation_matrix_from_dlt(dltcoefs[:,2])

m = np.column_stack((np.eye(3), np.zeros(3))) # the 3x4 matrix to 'grease the wheels'

P11kmt = np.matmul(Kteax, np.matmul(m,T1.T))
P22kmt = np.matmul(Kteax, np.matmul(m,T2.T))
P33kmt = np.matmul(Kteax, np.matmul(m,T3.T))

cam1 = Camera(1, [0,0,0], fx, px, py, fx, fy, Kteax, [0,0,0],
              np.eye(3), np.zeros(5), [0,0,0], P11kmt)
cam2 = Camera(2, [0,0,0], fx, px, py, fx, fy, Kteax, [0,0,0],
              np.eye(3), np.zeros(5), [0,0,0], P22kmt)
cam3 = Camera(2, [0,0,0], fx, px, py, fx, fy, Kteax, [0,0,0],
              np.eye(3), np.zeros(5), [0,0,0], P33kmt)

xyz_Tmat_based = []

for c1_pt, c2_pt in zip(cam1_undist[11:24], cam2_undist[11:24]):
    position_homog = cv2.triangulatePoints(P11kmt, P22kmt,
                                           c1_pt.flatten(), c2_pt.flatten())
    xyz_Tmat_based.append(cv2.convertPointsFromHomogeneous(position_homog.T))    

xyz_Tmat_based = np.array(xyz_Tmat_based).reshape(-1,3)

#%%
# Generate projection matrix from DLT coefficients
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is based on my attempts at trying to recreate the projection matrix
# :math:`P` directly from the DLT coefficients (again inspired by the Hedrick lab link)

def extract_P_from_dlt(dltcoefs):
    '''function which recreates the Projection matrix
    Many thanks to cyborg on Stack Overflow 
    (https://stackoverflow.com/a/71321626/4955732).
    This function is a modified version of 
    Julian Jandeleit for sharing his MATLAB code    
    '''
    dltcoefs = np.append(dltcoefs, 0)
    norm = np.linalg.norm(dltcoefs)
    dltcoefs = dltcoefs/norm

    P = dltcoefs.reshape(3,4)
    return P

def extract_P_from_dlt_v2(dltcoefs):
    '''No normalisation
    '''
    dltcoefs = np.append(dltcoefs, 1)
    dltcoefs = dltcoefs

    P = dltcoefs.reshape(3,4)
    return P

# generate projection matrix 
Pcam1 = extract_P_from_dlt(c1_dlt)
Pcam2 = extract_P_from_dlt(c2_dlt)

# Now get 3D positions using cv2.triangulatePositions
xyz_P_based = []
for pt1, pt2 in zip(cam1_undist[11:24,:], cam2_undist[11:24,:]):
    pt1_homog, pt2_homog = (X.reshape(1,1,2) for X in [pt1, pt2])
    position = cv2.triangulatePoints(Pcam1, Pcam2, pt1_homog, pt2_homog)
    final_xyz = cv2.convertPointsFromHomogeneous(position.T).flatten()
    xyz_P_based.append(final_xyz)

xyz_P_based = np.array(xyz_P_based).reshape(-1,3)

#%% 
# Visualise all of these points - we see the parabolic path of an object that's
# been thrown and is falling as it flies through the air.

plt.figure()
ax = plt.subplot(311, projection='3d')
ax.view_init(azim=1, elev=-54)
ax.plot(xyz_dlt_easywand[:,0],
        xyz_dlt_easywand[:,1],
        xyz_dlt_easywand[:,2], '*')
ax2 = plt.subplot(312, projection='3d')
ax2.view_init(azim=-59, elev=-54)
ax2.plot(xyz_Tmat_based[:,0], xyz_Tmat_based[:,1], xyz_Tmat_based[:,2], '*')
ax3 = plt.subplot(313, projection='3d')
ax3.view_init(azim=4, elev=-42)
ax3.plot(xyz_P_based[:,0], xyz_P_based[:,1], xyz_P_based[:,2], '*')

#%% 
# How similar or dissilimar are the points. They may have different origins and 
# axis orientations - but the euclidean distances between points should remain 
# the same

indices = np.arange(5,10)
pbased_distmat = get_point2point_matrix(xyz_P_based[indices,:])
tbased_distmat = get_point2point_matrix(xyz_Tmat_based[indices,:])
easywand_distmat = get_point2point_matrix(xyz_dlt_easywand[indices,:])

#%%
# Camera centres across the different methods
# DLT based method (code from Ty Hedrick)
Ccam1_dltmethod = cam_centre_from_dlt(dltcoefs[:,0])
Ccam2_dltmethod = cam_centre_from_dlt(dltcoefs[:,1])
intercam_dist_dlt = distance.euclidean(Ccam1_dltmethod, Ccam2_dltmethod)

#%% Projection matrix based method

Ccam1_Pmethod = get_cam_centre_from_projectionmat(Pcam1)
Ccam2_Pmethod = get_cam_centre_from_projectionmat(Pcam2)
intercam_dist_P = distance.euclidean(Ccam1_Pmethod, Ccam2_Pmethod)

print(f'Inter-camera centre distance: \n Projection mat: {intercam_dist_P}\n DLT method: {intercam_dist_dlt}')


#%% 
# Get the overall accelaration
acc_dlt = smooth_and_acc(xyz_dlt_easywand,fps=25)
acc_P = smooth_and_acc(xyz_P_based)




