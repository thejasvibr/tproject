'''
Troubleshooting
===============

2022-02-14
----------
I'm getting the hang of how 3D points are projected onto a cameras 2D sensor. Still
some questions remaining:
    
    * How does the pinhole camera model map to a standard digital camera lens+sensor setup?

Additionally, Right now I'm getting weird answers when I try to generate my own object and 
project onto 2D. The u,v coordinates don't seem to make sense somehow. 

Also, even the comments say that the 
'''
import cv2
import numpy as np 
import pandas as pd
import track2trajectory.match2dto3d as m2d3d
import track2trajectory.synthetic_data as syndata
import track2trajectory.projection as projection
np.random.seed(82319)

cam1, cam2 = syndata.generate_two_synthetic_cameras_version2()

x,y,z = [np.float32(each) for each in [[0,0,0], [8,9,10], [0,-1,1]]]
points_in_3d = pd.DataFrame(data={'x':x,'y':y,'z':z,'frame':np.tile(1, x.size),
                                  'oid':np.arange(x.size)})


cam1_2dpoints, cam1_fails = projection.project_to_2d_and_3d(points_in_3d,
                                                                 cam1)
cam2_2dpoints, cam2_fails  = projection.project_to_2d_and_3d(points_in_3d,
                                                                  cam2)
fundamatrix_cam2fromcam1 = projection.calcFundamentalMatrix(cam1, cam2)


index = 1
p1_cam1 = np.float32(cam1_2dpoints.loc[index,['x','y']].tolist())
p1_cam2 = np.float32(cam2_2dpoints.loc[index,['x','y']].tolist())

point1_undistort = cv2.undistortPoints(p1_cam1, cam1.i_mtrx, cam1.cof_mtrx,
                             P=cam1.i_mtrx)
point2_undistort = cv2.undistortPoints(p1_cam2, cam2.i_mtrx, cam2.cof_mtrx,
                             P=cam2.i_mtrx)


tri_res = cv2.triangulatePoints(cam1.cm_mtrx, cam2.cm_mtrx,
                                p1_cam1, p1_cam2)




xyza = projection.triangulate_points_in3D(p1_cam1, p1_cam2, cam1, cam2)


print(xyza, points_in_3d.loc[index,['x','y','z']])
