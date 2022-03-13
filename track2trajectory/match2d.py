# -*- coding: utf-8 -*-
"""
2D matching methods
===================
"""

import cv2
import numpy as np 
import pandas as pd
from scipy.spatial import distance
import track2trajectory
import track2trajectory.camera as camera
from track2trajectory import get_closest_points
from track2trajectory.projection import triangulate_points_in3D, project_to_2d
import tqdm
make_homog = track2trajectory.make_homog

def generate_2d_correspondences(camera1, camera2, cam1_2dpoints, cam2_2dpoints,
                                        fundamental_matrix, **kwargs):
    '''Performs 2-way epipolar line matching ('2-way authentication' in Giray's thesis).
    First the epipolar of a source point from Cam1 is projected onto Cam2. The closest points
    to the epipolar line are chosen. For each of the candidates on Cam2 (2nd projection),
    the epipolar is projected back onto Cam1. If there are >1 Cam1 points close to the line
    then the candidate is discarded. If there is only one candidate close the epipolar
    on Cam1 and it matches the source point - then it is considered a reliable correspondence.

    TODO
    ----
    * implement the fundamental matrix based distance. This method checks
    the residual in the x'Fx . In the ideal case, the product should be 0, 
    otherwise - the best correspondence will still give the lowest value.

    Parameters
    ----------
    camera1, camera2 : Camera.camera instances
    cam1_2dpoints, cam2_2dpoints : pd.DataFrame
        With columns: frame, x, y, oid, cid
    fundamental_matrix: (3,3) np.array
        The fundamental matrix mapping the points between cam1 to cam2.
    rel_threshold : float > 0
        The relative threshold used to decide how many of the points are close
        enough to the epipolar line. All points within  min_distance x (1+rel_threshold)
        are considered close. If the minimum distance is 0, then all points with
        0 distance are returned. For default see :code:`helper.get_closest`

    Returns
    -------
    correspondences : pd.DataFrame
        A dataframe with columns :code:`frame, c1_oid, c2_oid`.
        If no match was found for a point on camera 1 then it is assigned a
        'np.nan' value.
    failed_matches_cam1 : int 
        The number of points that couldn't be matched

    References
    ----------
    * Tandogan, Giray, 2022, 3D Trajectory Reconstruction for Animal Data, Master's
    thesis (Uni. Konstanz), page 9

    See Also
    --------
    projection.calcFundamentalMatrix
    match2d.find_candidates
    helper.get_closest_points
                                           
    '''
    backpred_tol = kwargs.get('backpred_tol', 10) # backprediction tolerance
    failed_matches_cam1 = 0
    correspondences = {}
    for rownum in tqdm.tqdm(list(cam1_2dpoints.index)):
        framenum = cam1_2dpoints.loc[rownum,'frame']
        
        at_frame_c1 = cam1_2dpoints[cam1_2dpoints['frame']==framenum]
        at_frame_c2 = cam2_2dpoints[cam2_2dpoints['frame']==framenum]
        if np.logical_or(at_frame_c1.empty, at_frame_c2.empty):
            failed_matches_cam1 += 1 
            continue

        source_point = at_frame_c1.loc[rownum,['x','y']].to_numpy(dtype='float32')
        c1oid = at_frame_c1.loc[rownum,['oid']].tolist()[0]
        # For each point on Cam1 - try to find corresponding point on cam2 with 2wayprojection
        cam2_best_points = find_candidates(at_frame_c2, fundamental_matrix,
                                           camera1, camera2, source_point, **kwargs)
        print(f'frame: {framenum} rownum:{rownum},\
              {c1oid},{source_point}\
              Cam2 best points: {cam2_best_points}')
        # If a single point on Cam2 is found then great, else leave this for now 
        # (and store the possible candidates?)
        if not len(cam2_best_points)==1:
            print(f'cam2 candidates neq 1: {cam2_best_points}')
            failed_matches_cam1 += 1 
        else:
            cam1_2way_points = find_candidates(at_frame_c1, fundamental_matrix,
                                           camera2, camera1, cam2_best_points[0],
                                                               **kwargs)
            if len(cam1_2way_points)!=1:
                print(f'cam1-2way neq 1: {cam1_2way_points}')
                correspondences[f'frame_{framenum}_c1_oid_{c1oid}'] = f'{np.nan}'
                same_point_check = False
                failed_matches_cam1 += 1 
            else:
                # check that cam1_2way_points is the same as the source point
                same_point_check = np.allclose(source_point,
                                               cam1_2way_points.flatten(),
                                               atol=backpred_tol)
                if not same_point_check:
                    print(f'back-match not close enough, {cam1_2way_points}')

            print(f'Cam2 best: {cam2_best_points[0][0],cam2_best_points[0][1]} \n')
            
            if same_point_check:
                x0, y0 = cam2_best_points[0][0], cam2_best_points[0][1]
                # get object ID of the cam 2 point
                same_x = at_frame_c2['x'].sub(x0).abs().idxmin()
                same_y = at_frame_c2['y'].sub(y0).abs().idxmin()
                if same_x==same_y:
                    cam2_on_frame_row = at_frame_c2.loc[same_x,:]
                    c2oid = cam2_on_frame_row['oid']
                    correspondences[f'frame_{framenum}_c1_oid_{c1oid}'] = f'c2_oid_{c2oid}'
                else:
                    raise ValueError(f'wtf :{same_x, same_y, at_frame_c2}')
            else:
                failed_matches_cam1 += 1

    correspondece_data = pd.DataFrame(columns=['frame','c1_oid','c2_oid'], 
                                      index=np.arange(len(correspondences)))
    
    for i,(key, value) in enumerate(correspondences.items()):
        parts = key.split('_')
        match = value.split('_')
        correspondece_data.loc[i, 'frame'] = parts[1]
        correspondece_data.loc[i, 'c1_oid'] = parts[-1]
        correspondece_data.loc[i, 'c2_oid'] = match[-1]
    
    # Run check to see if there are repeat assignments of Cam 2 points
       
    return correspondece_data, failed_matches_cam1

def find_candidates(all_2d_points_camera2, fund_matrix, camera1: camera.Camera,
                   camera2: camera.Camera, source_point, **kwargs):
    '''
   
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
   rel_threshold : float > 0
       The absolute relative threshold for point distances. Defaults to 0.1.
       This means if distances are [0.1, 1, 2, 0.11, 0.105], then all 
       distances <= :math:`minimum_value \times (1+rel_threshold)` will be 
       considered close.
   match_method : str
       One of ('epipolar', 'fundmat')
    
   Returns 
   -------
   closest_points : (N,2) np.array
       xy coordinates of the closest point(s) to the epipolar line

   See Also
   --------
   projection.calcFundamentalMatrix
   helper.get_closest_points
   
   Notes
   -----
   This if the refactored version of the original find_candidate
    '''
    match_method = kwargs.get('match_method', 'epipolar')
    if match_method == 'epipolar':
        closest_points  = find_candidates_epipolar(all_2d_points_camera2,
                                                  fund_matrix, camera1,
                                                  camera2, source_point,
                                                  **kwargs)
    elif match_method == 'fundmat':
        closest_points  = find_candidates_fundmat(all_2d_points_camera2, fund_matrix,
                                                  camera1, camera2,
                                                  source_point, **kwargs)
    elif match_method == '3dbackpred':
        closest_points = find_candidates_backproj(all_2d_points_camera2, 
                                                  fund_matrix,
                                                  camera1,
                                                  camera2,
                                                  source_point, **kwargs)
    else: 
        raise ValueError(f'Given match method {match_method} is invalid.\
                         The only valid match methods are epipolar or fundmat')
    return closest_points
        
def get_epiline(all_2d_points_camera2, fund_matrix,
                camera1, camera2, source_point):
    '''
    '''
    # Set camera ids for computeCorrespondEpilines()
    if camera1.id < camera2.id:
        cid = 1
    else:
        cid = 2
   
    if np.sum(np.isnan(source_point))>1:
        return np.array([np.nan, np.nan])
    
    # Get epiline from cam1 to cam2
    epilines = cv2.computeCorrespondEpilines(source_point.reshape(-1,2),
                                             cid, np.float32(fund_matrix))
    return epilines
        
def find_candidates_epipolar(all_2d_points_camera2, fund_matrix,camera1: camera.Camera,
                   camera2: camera.Camera, source_point, **kwargs):
    '''
    Gives the 2D points closest to the epipolar line drawn on camera 2. The epipolar
    line is derived from the candidate point on camera 1. 

     If there are >1 points that are similarly close to the epipolar line, then 
     multiple candidates are returned.

    '''
    epilines = get_epiline(all_2d_points_camera2, fund_matrix,
                           camera1, camera2, source_point)
    a,b,c = epilines[0][0]
    points_xy = all_2d_points_camera2.loc[:,['x','y']].to_numpy(dtype='float32')    
    # if point lies close the line the ax+by+c will be close to 0
    # if it lies on the line, then the sum will be 0
    distances = []
    for each in points_xy:
        x,y = each
        distances.append(abs(a*x+b*y+c))
    closest_points = get_closest_points(points_xy, distances, **kwargs)
    return closest_points 
    

def find_candidates_fundmat(all_2d_points_camera2, fund_matrix,camera1: camera.Camera,
                   camera2: camera.Camera, source_point, **kwargs):
    '''
    Uses the residual in the x'Fx relation to detect the best correspondence
    between points. 
    
    References
    ----------
    * Hartley & Zisserman, page 246
    
    '''
    xy_pts_cam2 = all_2d_points_camera2.loc[:,['x','y']].to_numpy(dtype=np.float32)
    residuals = []
    for each_pt in xy_pts_cam2:
        residual = check_xprimeFx(fund_matrix, source_point, each_pt, camera1, camera2)
        residuals.append(residual)
    closest_points = get_closest_points(xy_pts_cam2, residuals, **kwargs)
    return closest_points
    

def find_candidates_backproj(all_2d_points_camera2, fund_matrix,camera1: camera.Camera,
                   camera2: camera.Camera, source_point, **kwargs):
    '''
    camera1 is the 'source' camera where the source point is located
    
    A 3D point is generated first
    '''
    backpred_tol = kwargs.get('backpred_tol', 10)
    xy_pts_cam2 = all_2d_points_camera2.loc[:,['x','y']].to_numpy(dtype=np.float32)
    all_distances = []
    for each_cam2 in xy_pts_cam2:
        threedpoint = triangulate_points_in3D(source_point, each_cam2,
                                              camera1, camera2)
        xy_proj, failed = project_to_2d(threedpoint, camera1)
        if failed:
            all_distances.append(np.nan)
            continue
        # check that the 2D projection matches the source 2D point
        if np.allclose(xy_proj, source_point, atol=backpred_tol):
            proj_distance = distance.euclidean(xy_proj, source_point)
            all_distances.append(proj_distance)
        else:
            all_distances.append(np.nan)
    closest_points = get_closest_points(xy_pts_cam2, all_distances, **kwargs)
    return closest_points
    
        


def check_xprimeFx(F,x, xprime, camera1, camera2):
    if camera1.id < camera2.id:
        Fx = np.matmul(F, make_homog(x))
        xprimeFx = np.matmul(make_homog(xprime).T, Fx)
    else:
        Fx = np.matmul(F.T, make_homog(x))
        xprimeFx = np.matmul(make_homog(xprime).T, Fx)
    return xprimeFx

# def find_candidate(all_2d_points, fund_matrix, camera1: camera.Camera,
#                    camera2: camera.Camera, candidate_point):
#     '''
#     Gives the 2D point closest to the epipolar line drawn on camera 2. The epipolar
#     line is derived from the candidate point on camera 1. 

#     Parameters
#     ----------
#     all_2d_points : pd.DataFrame
#         xy coordinates for on the camera 'reference' camera  (assigned to number 1)
#         With at the least the following columns : x, y, oid, frame
#     fund_matrix: 3x3 np.array
#         The fundamental matrix. The fundamental matrix is a matrix which transforms
#         a set of points Xi on  camera 1 onto their corresponding points Xj onto 
#         camera 2.
#     camera1, camera2 : camera.Camera instances
#     candidate_point :  np.array (np.float32)

#     Returns 
#     -------
#     pre_p : np.array
#         x,y coordinates of the point with min distance to epipolar line on the 
#         other camera
#     pre_min : float
#         The perpendicular Euclidean distance of the best match to the epipolar line
#     pre_frame : int 
#         Frame number. This is here more for diagnostic reasons as the pre_frame should 
#         match the same frame number as all_2d_points_c1
#     pre_id : int
#         Object id of the best point in the 'other' camera 
#     row_num : int 
#         The row number of the best matching point in all_2d_points_c1. This too is here more 
#         for diagnostic reasons.

#     See Also
#     --------
#     projection.calcFundamentalMatrix

#     TODO
#     ----
#     Remove the redundancy in output objects
#     '''
#     run_len = len(all_2d_points)
#     pre_min = math.inf
#     x_cand, y_cand, pre_frame, pre_id = -1, -1, -1, -1
#     pre_p = []
#     row_num = -1

#     for k in range(run_len):
#         current_point_cam1 = np.float32([all_2d_points.iloc[k]['x'],
#                                          all_2d_points.iloc[k]['y']])
#         if len(candidate_point) != 0:
#             points_undistorted_2 = cv2.undistortPoints(candidate_point, 
#                                                        camera2.i_mtrx,
#                                                        camera2.cof_mtrx,
#                                                        P=camera2.i_mtrx)
#         else:
#             return [], -1, -1, -1, -1

#         # Set camera ids for computeCorrespondEpilines()
#         if camera1.id < camera2.id:
#             cid = 1
#         else:
#             cid = 2

#         # Find epipolar line values: a,b,c. From ax+by+c=0.

#         df1numpya = np.float64(all_2d_points[['x', 'y']].to_numpy())
#         epilines = cv2.computeCorrespondEpilines(df1numpya, cid, fund_matrix)
#         v_a = epilines[k][0][0]
#         v_b = epilines[k][0][1]
#         v_c = epilines[k][0][2]
#         dist_fund_point = (points_undistorted_2[0][0][0] * v_a) + (points_undistorted_2[0][0][1] * v_b) + v_c

#         if abs(dist_fund_point) < pre_min:
#             pre_min = abs(dist_fund_point)
#             pre_id = all_2d_points.iloc[k]['oid']
#             pre_frame = all_2d_points.iloc[k]['frame']
#             pre_p = current_point_cam1
#             row_num = k
#         else:
#             pass

#     return pre_p, pre_min, pre_frame, pre_id, row_num

