# -*- coding: utf-8 -*-
"""
Re-writing the match3d function
================================





@author: Thejasvi
"""
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from scipy.spatial import distance
from scipy.spatial.transform import Rotation
import track2trajectory
import track2trajectory.camera as camera
import tqdm
import datetime as dt
from track2trajectory.prediction import kalman_predict
from track2trajectory.projection import triangulate_points_in3D, project_to_2d_and_3d
from track2trajectory.projection import calcFundamentalMatrix
import track2trajectory.match2d as match2d
import track2trajectory.synthetic_data as syndata
from track2trajectory.match2d import generate_2d_correspondences
from track2trajectory.match3d import get_3d_positions_from2dmatches

make_homog = track2trajectory.make_homog
np.random.seed(90)

def make_2dprojection_data(cam1, cam2, points_in_3d):
    
    cam1_2dpoints, cam1_fails = project_to_2d_and_3d(points_in_3d,
                                                                     cam1, mu=0,
                                                                     sigma=1e-4)
    cam2_2dpoints, cam2_fails  = project_to_2d_and_3d(points_in_3d,
                                                                      cam2,
                                                                      mu=0,
                                                                      sigma=1e-4)
    all_2dpoints = pd.concat([cam1_2dpoints,
                              cam2_2dpoints]).reset_index(drop=True)
    return all_2dpoints

# rotate cam2 up a bit and towards the origin
cam2_rotation = Rotation.from_euler('xyz', [0,0,0],degrees=True).as_matrix()
cam2_centre = [0,2,0]
cam1, cam2 = syndata.generate_two_synthetic_cameras_version2(cam2_Rotation=cam2_rotation,
                                                             cam2_C=cam2_centre)

def get_camC(cam):
    rr = cam.cm_mtrx[:3,:3]
    tt = cam.cm_mtrx[:,-1]
    return -np.matmul(rr.T, tt)

# raise ValueError('Something wrong with cam2 centre assignment!!!')

x_range = np.linspace(-0.5,0.5,2)
y_range = np.linspace(7,9,2)
z_range = np.linspace(-0.5,0.5,2)

num_particles = 15
num_frames = 20
points_in_3d = syndata.make_brownian_particles(num_particles,
                                               [[-5,5],
                                                   [7,10],
                                                   [-1,1]], frames=num_frames,
                                               stepsize=0.2)

all_2dpoints = make_2dprojection_data(cam1, cam2, points_in_3d)
fundamatrix_cam2fromcam1 = calcFundamentalMatrix(cam1, cam2)

camera1, camera2 = cam1, cam2
all_2d_points = all_2dpoints.copy()
all_3d_groundtruth = points_in_3d.copy()
result_file_name = 'miaomiaow'
fundamental_matrix = fundamatrix_cam2fromcam1.copy()

cam1_2dpoints = all_2d_points[all_2d_points['cid']==0].reset_index(drop=True)
cam2_2dpoints = all_2d_points[all_2d_points['cid']==1].reset_index(drop=True)
# alter the camera 2 oids just to check that it's not specific to a numerical oid
import string

oid_replacement_dict = {each: string.ascii_lowercase[each]  for each in range(num_particles)}
cam2_2dpoints['oid'] = cam2_2dpoints['oid'].replace(oid_replacement_dict)

#%% Broad strokes of the function 
cam_matches, fails = generate_2d_correspondences(camera1, camera2, cam1_2dpoints,
                                                    cam2_2dpoints, fundamental_matrix,
                            )

cam3d = get_3d_positions_from2dmatches(cam_matches, cam1_2dpoints,
                                       cam2_2dpoints, camera1, camera2)

traj3d = cam3d.copy()
traj3d['traj_id'] = traj3d['c1_oid'].astype('str') +'_'+ traj3d['c2_oid'].astype('str')
unique_trajids = set(traj3d['traj_id'])
# remove all traj ids with nan in them 
nonan_trajids = [each for each in unique_trajids if not 'nan' in each]

# If Kalman Filtering is possible, run the KF to predict the position from a set
# of points that are paired

# If KF is not possible, generate the 2D projection of the predicted 3D point, and

#%%

import matplotlib.pyplot as plt
import matplotlib.animation as animation



fig = plt.figure()

ax = plt.subplot(121)
ax.imshow(np.random.normal(0,1,1920*1040).reshape(1040,1920), origin='upper')
ax2 = plt.subplot(122)
ax2.imshow(np.random.normal(0,1,1920*1040).reshape(1040,1920),origin='upper')


cam1_lines = [ ax.plot([], [], '-', lw=1)[0] for each in range(num_particles)]
cam2_lines = [ ax2.plot([], [], '-', lw=1)[0] for each in range(num_particles)]

def update(i):
    # camera 1
    ax.set_title(f'Cam1 Frame {i}')
    for j,each_particle in enumerate(cam1_lines):
        thispoint_df = cam1_2dpoints[cam1_2dpoints['oid']==j].reset_index(drop=True)
        try:
            each_particle.set_data(thispoint_df.loc[i-10:i,['x']],
                                   thispoint_df.loc[i-10:i,['y']])
        except:
            each_particle.set_data(thispoint_df.loc[:i,['x']],thispoint_df.loc[:i,['y']])

    # camera 2
    ax2.set_title(f'Cam2 Frame {i}')
    for j,each_particle2 in enumerate(cam2_lines):
        alphabet = string.ascii_lowercase[j]
        thispoint_df = cam2_2dpoints[cam2_2dpoints['oid']==alphabet].reset_index(drop=True)
        try:
            each_particle2.set_data(thispoint_df.loc[i-10:i,['x']],
                                   thispoint_df.loc[i-10:i,['y']])
        except:
            each_particle2.set_data(thispoint_df.loc[:i,['x']],
                                   thispoint_df.loc[:i,['y']])



# Creating the Animation object
ani = animation.FuncAnimation(fig, update, num_frames, interval=40, repeat=False);
#writergif = animation.PillowWriter(fps=25) 
#ani.save('2D_points.gif', writer = writergif)
# plt.figure()
# plt.subplot(121)
# plt.imshow(np.random.normal(0,0.01,1920*1040).reshape(-1,1920))
# for i,(x,y) in enumerate(zip(at_frame_c1['x'],at_frame_c1['y'])):
#     plt.plot(x,y,'*')
#     plt.text(x,y,str(i))
# plt.subplot(122)
# plt.imshow(np.random.normal(0,0.01,1920*1040).reshape(-1,1920))
# for i,(x,y) in enumerate(zip(at_frame_c2['x'],at_frame_c2['y'])):
#     plt.plot(x,y,'*')
#     plt.text(x,y,str(i+7))
