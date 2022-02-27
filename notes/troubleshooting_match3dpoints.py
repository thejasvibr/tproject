# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 13:09:20 2022

@author: theja
"""
import cv2
import datetime as dt
import numpy as np
import pandas as pd
import track2trajectory.camera as camera
import track2trajectory.projection as projection

import track2trajectory.synthetic_data as syndata
import tqdm
from scipy.spatial.transform import Rotation

pickrandom = lambda X, num: np.random.choice(X,num,replace=False)


cam2_rotation = Rotation.from_euler('xyz', [0,20,0],degrees=True).as_matrix()

cam1, cam2 = syndata.generate_two_synthetic_cameras_version2(cam2_Rotation=cam2_rotation)


# x_range = np.linspace(-0.5,0.5,10) 
# y_range = np.linspace(10,20,10)
# z_range = np.linspace(-2,2,10)

# threed_data = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)

threed_data = np.column_stack((np.random.normal(0,0.2,8),
                               np.linspace(10,20,8),
                               np.linspace(-1,1,8)))



points_in_3d = pd.DataFrame(data={'x':threed_data[:,0],
                                       'y':threed_data[:,1],
                                       'z':threed_data[:,2],
                                       'frame':np.tile(1, threed_data.shape[0]),
                                       'oid':np.arange(threed_data.shape[0])})

cam1_2dpoints, cam1_fails = projection.project_to_2d_and_3d(points_in_3d,
                                                                 cam1)
cam2_2dpoints, cam2_fails  = projection.project_to_2d_and_3d(points_in_3d,
                                                                  cam2)
all_2dpoints = pd.concat([cam1_2dpoints,
                          cam2_2dpoints]).reset_index(drop=True)

fundamatrix_cam2fromcam1 = projection.calcFundamentalMatrix(cam1,
                                                                 cam2)
#%% 
# Find closest match for point on cam1 
cam2point = cam2_2dpoints.loc[:,['x','y']].to_numpy(dtype='float32')
cam1point = cam1_2dpoints.loc[:,['x','y']].to_numpy(dtype='float32')
epilines = cv2.computeCorrespondEpilines(cam2point,
                                         2, fundamatrix_cam2fromcam1)

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(np.random.normal(0,0.0001,1920*1040).reshape(1040,1920),
           cmap='gray_r')
r,cols = 1040,1920
for i,point in enumerate(cam1point):
    a,b,c = epilines[i].flatten()
    x0, y0 = 0, -c/b
    x1 = cols
    y1 = (-c - a*x1)/b

    plt.plot(point[0],point[1],'*')
    plt.plot([x0,x1],[y0,y1],'-')



