# -*- coding: utf-8 -*-
"""
2D matching methods
===================
"""

import cv2
import math
import numpy as np 
from scipy.spatial import distance
import track2trajectory.camera as camera


def find_candidate_v2(all_2d_points_camera2, fund_matrix, camera1: camera.Camera,
                   camera2: camera.Camera, source_point):
    '''
    fund_matrix:  (3,3) np.array
        Fundamental matrix projects points from cam1 -> cam2
    
    
    Notes 
    -----
    p_x, p_y
    epi_lines : ax + by + c 
        or , y = -(ax+c)/b
        
    
    '''
    # Set camera ids for computeCorrespondEpilines()
    if camera1.id < camera2.id:
        cid = 1
    else:
        cid = 2

    # Get epiline from cam1 to cam2
    epilines = cv2.computeCorrespondEpilines(source_point, cid, fund_matrix)
    points_xy = all_2d_points_camera2.loc[:,['x','y']].to_numpy(dtype='float32')    
    a,b,c = epilines[0][0]
    x1y1, x2y2 = get_end_points_of_line(a,b,c, camera2.c_x*2)
    print(f'x1y1, x2y2: {x1y1, x2y2}')
    # orthogonal distance from the epipolar line for each point on camera2
    orth_distances = [orthogonal_distance(x1y1, x2y2, each_point)  for each_point in points_xy]

    closest_point = points_xy[np.argmin(orth_distances),:]
    # closest_point = get_closest_point_to_epiline(points_xy, orth_distances,
    #                                              fund_matrix)
    return closest_point, orth_distances



def find_candidate(all_2d_points, fund_matrix, camera1: camera.Camera,
                   camera2: camera.Camera, candidate_point):
    '''
    Gives the 2D point closest to the epipolar line drawn on camera 2. The epipolar
    line is derived from the candidate point on camera 1. 

    Parameters
    ----------
    all_2d_points : pd.DataFrame
        xy coordinates for on the camera 'reference' camera  (assigned to number 1)
        With at the least the following columns : x, y, oid, frame
    fund_matrix: 3x3 np.array
        The fundamental matrix. The fundamental matrix is a matrix which transforms
        a set of points Xi on  camera 1 onto their corresponding points Xj onto 
        camera 2.
    camera1, camera2 : camera.Camera instances
    candidate_point :  np.array (np.float32)

    Returns 
    -------
    pre_p : np.array
        x,y coordinates of the point with min distance to epipolar line on the 
        other camera
    pre_min : float
        The perpendicular Euclidean distance of the best match to the epipolar line
    pre_frame : int 
        Frame number. This is here more for diagnostic reasons as the pre_frame should 
        match the same frame number as all_2d_points_c1
    pre_id : int
        Object id of the best point in the 'other' camera 
    row_num : int 
        The row number of the best matching point in all_2d_points_c1. This too is here more 
        for diagnostic reasons.

    See Also
    --------
    projection.calcFundamentalMatrix

    TODO
    ----
    Remove the redundancy in output objects
    '''
    run_len = len(all_2d_points)
    pre_min = math.inf
    x_cand, y_cand, pre_frame, pre_id = -1, -1, -1, -1
    pre_p = []
    row_num = -1

    for k in range(run_len):
        current_point_cam1 = np.float32([all_2d_points.iloc[k]['x'],
                                         all_2d_points.iloc[k]['y']])
        if len(candidate_point) != 0:
            points_undistorted_2 = cv2.undistortPoints(candidate_point, 
                                                       camera2.i_mtrx,
                                                       camera2.cof_mtrx,
                                                       P=camera2.i_mtrx)
        else:
            return [], -1, -1, -1, -1

        # Set camera ids for computeCorrespondEpilines()
        if camera1.id < camera2.id:
            cid = 1
        else:
            cid = 2

        # Find epipolar line values: a,b,c. From ax+by+c=0.

        df1numpya = np.float64(all_2d_points[['x', 'y']].to_numpy())
        epilines = cv2.computeCorrespondEpilines(df1numpya, cid, fund_matrix)
        v_a = epilines[k][0][0]
        v_b = epilines[k][0][1]
        v_c = epilines[k][0][2]
        dist_fund_point = (points_undistorted_2[0][0][0] * v_a) + (points_undistorted_2[0][0][1] * v_b) + v_c

        if abs(dist_fund_point) < pre_min:
            pre_min = abs(dist_fund_point)
            pre_id = all_2d_points.iloc[k]['oid']
            pre_frame = all_2d_points.iloc[k]['frame']
            pre_p = current_point_cam1
            row_num = k
        else:
            pass

    return pre_p, pre_min, pre_frame, pre_id, row_num

def get_closest_point_to_epiline(source_point, other_points, dist_to_epiline,
                                 fundamental_matrix,
                                 **kwargs):
    '''
    Check for points with very similar orthogonal distances
    
    if multiple points lie close to the epipolar line - check that the 
    fundamental X'.t F X = 0   matrix relation is satisfied. Choose the point
    which yields the result closest to zero.
    '''
    rel_threshold = kwargs.get('relative_threshold', 0.1)
    # get relative distances 
    rel_distances = np.array(dist_to_epiline)/np.min(dist_to_epiline)
    points_within_limits = np.logical_and(rel_distances<1+rel_threshold, 
                                          rel_distances>1-rel_threshold)
    num_fairlyclose_points = np.sum(points_within_limits)
    if num_fairlyclose_points==1:
        closest_point = other_points[np.argmin(dist_to_epiline),:]
        return closest_point
    # when there are >1 close points
    close_points = other_points[points_within_limits,:]
    # check how well the fundamental matrix relation is satisfied
    fundmat_residual = []
    source_point_homog = np.concatenate((source_point,np.array[1])).flatten()
    for each in close_points:
        each_homog = np.concatenate((each, np.array([1]))).flatten()
        Fx = np.matmul(fundamental_matrix, source_point)
        residual = np.matmul(each_homog.T, Fx)
        fundmat_residual.append(residual)
    
    
    

def get_end_points_of_line(a,b,c, x_max):
    '''
    ax + by + c = 0

    by = -ax -c

    y = (-ax - c)/b

    y = -(ax+c)/b
    '''
    x1, y1 = 0, 0 
    x2 = x_max
    y2 = - (a*x2 + c)/b

    x1y1 = np.float32([x1, y1])
    x2y2 = np.float32([x2, y2])
    return x1y1, x2y2
    

def orthogonal_distance(point1, point2, point3):
    '''
    Point1,2 define the ends of the line, while point3 is the candidate
    to which we want to calculate the orthogonal distance for. 
    
    Orthogonal distance between point 3 (P3) and the line between P1,2 is
    given by the distance between (x,y) and P3
    
    x = x1 + u (x2-x1)

    y = y1 + u (y2-y1)
    
    where u is defined as :math:`u=\\frac{(x3-x1)(x2-x1)+(y3-y1)(y2-y1)}{||p2-p1||^{2}}`

    The
    
    References
    ----------
    * Minimum distance between a Point and a Line, 
    http://paulbourke.net/geometry/pointlineplane/ accessed 2022-02-22

    '''
    point1, point2, point3 = [np.array(each) for each in [point1, point2, point3]]

    x1,y1 = point1
    x2,y2 = point2
    x3,y3 = point3

    u_num = (x3-x1)*(x2-x1) + (y3-y1)*(y2-y1)
    u_denom = np.linalg.norm(point2-point1)**2
    u = u_num/u_denom

    x = x1 + u*(x2-x1)
    y = y1 + u*(y2-y1)

    orthogonal_distance = distance.euclidean([x,y], point3)
    return orthogonal_distance


if __name__ == '__main__':
    import pandas as pd
    import track2trajectory.projection as projection
    import track2trajectory.synthetic_data as syndata
    from scipy.spatial.transform import Rotation
    np.random.seed(5)

    pickrandom = lambda X, num: np.random.choice(X,num,replace=False)
    
    cam1_R = Rotation.from_euler('xyz', [0,10,0],degrees=True).as_matrix()
    cam1, cam2 = syndata.generate_two_synthetic_cameras_version2(cam1_Rotation=cam1_R)

    x_range = pickrandom(np.linspace(-0.5,0.5,10), 2)
    y_range = pickrandom(np.linspace(5,10,10),2)
    z_range = pickrandom(np.linspace(-1,1,10),2)
    
    threed_data = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)
    threed_data += np.random.normal(0,0.2,threed_data.size).reshape(-1,3)
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
    
    index = 2
    source_point = cam1_2dpoints.loc[index,['x','y']].to_numpy(dtype='float32').reshape(-1,2)
    output, dists = find_candidate_v2(cam2_2dpoints, fundamatrix_cam2fromcam1, 
                      cam1, cam2, source_point)
    print(source_point, output, cam2_2dpoints.loc[index,['x','y']])
