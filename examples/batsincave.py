# -*- coding: utf-8 -*-
"""
Matching and tracking bats in a cave
====================================
Here let's use some more experimental data to find 2D correspondences, 
and finally 3D paths for 2-3 bats seen flying together in a cave. 
"""
import matplotlib.pyplot as plt 
import mat73
import numpy as np 
import pandas as pd
from track2trajectory.projection import calcFundamentalMatrix
from track2trajectory.match3d import match_2dpoints_to_3dtrajectories
from track2trajectory.match2d import generate_2d_correspondences
from track2trajectory.camera import Camera
from track2trajectory.projection import project_to_2d
from track2trajectory.synthetic_data import make_rotation_mat_fromworld, get_cam_coods
from track2trajectory.dlt_to_world import transformation_matrix_from_dlt, cam_centre_from_dlt

# sphinx_gallery_thumbnail_path = '_static/bats3dtraj.png'

threecam_xy = pd.read_csv('DLTdv8_data_p000_15000_3camxypts.csv')

def reformat_dltdv_frame(dltdv_df, camname, camnum):
    '''
    '''
    cam_cols = [dltdv_df.loc[:,column] for column in dltdv_df.columns if camname in column]
    horizontal_concat = pd.concat(cam_cols,1)
    unique_points = np.unique([each.split('_')[0] for each in horizontal_concat.columns])
    
    point_subdfs = []
    for point in unique_points:
        ptx = point + '_'+camname+'_X'
        pty = point + '_'+camname+'_Y'
        subdf = dltdv_df.loc[:,[ptx,pty]]
        subdf = subdf.rename(columns={ptx: "x", pty: "y"})
        subdf['frame'] = np.arange(dltdv_df.shape[0])
        subdf['oid'] = camname +'-'+point
        subdf['cid'] = camnum
        point_subdfs.append(subdf)
    formatted_df = pd.concat(point_subdfs,axis=0).reset_index(drop=True)
    return formatted_df
        
cam1_xy = reformat_dltdv_frame(threecam_xy, 'cam1', 1)
cam2_xy = reformat_dltdv_frame(threecam_xy, 'cam2', 2)
cam3_xy = reformat_dltdv_frame(threecam_xy, 'cam3', 3)

#%%
# camera image is 640 x 512
px,py = 320, 256
fx, fy = 526, 526 # in pixels

Kteax = np.array([[fx, 0, px],
                  [0, fy, py],
                  [0, 0,  1]])
# distortion coefficients
p1, p2 = np.float32([0,0]) # tangential distortion
k1, k2, k3 = np.float32([-0.3069, 0.1134, 0]) # radial distortion
dist_coefs = np.array([k1, k2, p1, p2, k3]) #in the opencv format

#%% All cameras are assumed to have the same distortion, and so we'll apply the 
# same undistortion to all of them. 
fname = '2018-08-17_wand_dvProject.mat'

matfile = False
if matfile:
    
    data_dict = mat73.loadmat(fname)
    dltcoefs = data_dict['udExport']['data']['dltcoef']
else:
    dltcoefs = pd.read_csv('2018-08-17_wand_dltCoefs.csv', header=None).to_numpy()
    

c1_dlt, c2_dlt, c3_dlt = [dltcoefs[:,col] for col in [0,1,2]]

def extract_P_from_dlt_v2(dltcoefs):
    '''No normalisation
    '''
    dltcoefs = np.append(dltcoefs, 1)
    dltcoefs = dltcoefs

    P = dltcoefs.reshape(3,4)
    return P

# generate projection matrix 
Pcam1 = extract_P_from_dlt_v2(c1_dlt)
Pcam2 = extract_P_from_dlt_v2(c2_dlt)
Pcam3 = extract_P_from_dlt_v2(c3_dlt)

# also get the rotation and translation data - placed in the cam->world frame

def make_transformation_matrix_from_dlt(dltc):
    '''
    '''
    

    t11, z11, _ = transformation_matrix_from_dlt(dltc)   
    shifter_mat = np.row_stack(([1,0,0,0],
                                [0,1,0,0],
                                [0,0,-1,0],
                                [0,0,0,1]))
    shifted_rotmat1 = np.matmul(t11, shifter_mat)[:3,:3]
    Tmat = make_rotation_mat_fromworld(shifted_rotmat1, t11[-1,:3])
    return Tmat

T1cam = make_transformation_matrix_from_dlt(c1_dlt)
T2cam = make_transformation_matrix_from_dlt(c2_dlt)
T3cam = make_transformation_matrix_from_dlt(c3_dlt)

cam1C = cam_centre_from_dlt(c1_dlt)
cam2C = cam_centre_from_dlt(c2_dlt)
cam3C = cam_centre_from_dlt(c3_dlt)

#%% 
# Generate camera objects and also remember to get the relevant rotation and 
# translation matrices in the camera->world frame. 


c_x, c_y, f, f_x, f_y = 320, 256, 526, 526, 526
cam1 = Camera(1, [0,0,0], 526, c_x, c_y, f_x, f_y, Kteax, T1cam[:3,-1], T1cam[:3,:3],
              dist_coefs, [], Pcam1)
cam2 = Camera(2, [0,0,0], 526, c_x, c_y, f_x, f_y, Kteax, T2cam[:3,-1], T2cam[:3,:3],
              dist_coefs, [], Pcam2)
cam3 = Camera(3, [0,0,0], 526, c_x, c_y, f_x, f_y, Kteax, T3cam[:3,-1], T3cam[:3,:3],
              dist_coefs, [], Pcam3)

F13 = calcFundamentalMatrix(cam1, cam3)
F12 = calcFundamentalMatrix(cam1, cam2)
F23 = calcFundamentalMatrix(cam2, cam3)
#%%

threed_matches = match_2dpoints_to_3dtrajectories(cam1, cam2, 
                                                  cam1_xy,
                                                  cam2_xy, F12,match_method='3dbackpred')

#%%
# Compare the readings with the groundtruth known

df = pd.read_csv('DLTdv8_data_p000_15000_3camxyzpts.csv')

pt3d_bycam = threed_matches.groupby('c1_oid')
plt.figure()
a0 = plt.subplot(111 ,projection='3d')
a0.view_init(azim=108, elev=21)
for names in ['cam1-pt1', 'cam1-pt2', 'cam1-pt3']:
    p1 = pt3d_bycam.get_group(names)   
    plt.plot(p1['x'],p1['z'],p1['y'],'-.',label=names)

for i,each in enumerate([cam1C, cam2C, cam3C]):
    a,b,c = each
    plt.plot(a,b,c,'^',label='cam'+str(i))
plt.legend()

plt.savefig('../docs/source/_static/bats3dtraj.png')
#%%
#.. image:: ../_static/bats3dtraj.png
#  :width: 400
