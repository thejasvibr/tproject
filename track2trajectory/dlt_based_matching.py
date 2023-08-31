import numpy as np 
import pandas as pd
from itertools import product
from track2trajectory.dlt_to_world import dlt_reconstruct_v2, dlt_inverse

def make_point_combis(focalpoint, pointscam2, pointscam3):
    '''
    Combine the focalpoint with all other points from other cameras.

    Parameters
    ----------
    focalpoint :(2,) np.array
        x,y coordinates (bottom left origin)
    pointscam2, pointscam3  : (Mpoints, 2) np.array
    
    Returns
    -------
    allpoint_combis : (MxN, 4) np.array
        Where camera 2 has M points and camera 3 has N points

    '''
    pointsexist = np.array([pointscam2.size>0, pointscam3.size>0])
    combi_ids = []
    allpoint_combis = []
    if np.all(pointsexist):
        index_combis = product(*[range(pointscam2.shape[0]), range(pointscam3.shape[0])])
        for (a,b) in index_combis:
            allpoint_combis.append(np.concatenate((focalpoint, pointscam2[a,:], pointscam3[b,:])))
            combi_ids.append([a,b])
    elif np.all(pointsexist == np.array([True, False])):
        index_combis = range(pointscam2.shape[0])
        allpoint_combis = []
        for a in index_combis:
            allpoint_combis.append(np.concatenate((focalpoint, pointscam2[a,:], np.tile(np.nan, 2) )))
            combi_ids.append([a,np.nan])
    elif np.all(pointsexist == np.array([False, True])):
        index_combis = range(pointscam3.shape[0])
        allpoint_combis = []
        for a in index_combis:
            allpoint_combis.append(np.concatenate((focalpoint, np.tile(np.nan, 2), pointscam3[a,:],  )))
            combi_ids.append([np.nan, a])
    return np.row_stack(allpoint_combis), combi_ids

def row_wise_dlt_reconstruct(uv_coods, dltcoefs, ):
    '''
    Performs DLT reconstruction on a row of x-y pixel coordinates 
    with bottom-left origin. 
    
    Parameters
    ----------
    uv_coods : (Npoints, Mcams*2) np.array
        x,y  coordinates. When object is not detected on that camera - a np.nan
        value must be inserted.
    dltcoefs : (11, Mcams) np.array
    
    Returns
    -------
    xyzpos : (3,) np.array
        Will output np.nan's if there is not enough cameras to reconstruct.

    '''
    # check which cameras have object detections 
    valid_coods_inds = np.where(np.invert(np.isnan(uv_coods)))[0]
    valid_uv = uv_coods[valid_coods_inds]
    valid_cams = np.int32(valid_coods_inds[0::2]/2)
    if len(valid_cams)>=2:
        valid_dlts = dltcoefs[:,valid_cams]
        xyzpos = dlt_reconstruct_v2(valid_dlts, valid_uv).flatten()
    else:
        xyzpos = np.tile(np.nan, 3)
    return xyzpos

def row_wise_dlt_inverse(xyzpoint, dltcoefs):
    '''
    Parameters
    ----------
    xyzpoint :  (3,) np.array
        XYZ coordinate of object
    dltcoefs : (11,Ncam) np.array
        DLT coefficients, column-wise per camera
    
    Returns
    -------
    output_uv : (Ncam*2,) np.array
        xy coordinates for each of the cameras. 
    '''
    if np.isnan(xyzpoint).sum()>0:
        return np.tile(np.nan, dltcoefs.shape[1]*2)
    else:
        output_uv = np.zeros(dltcoefs.shape[1]*2)
        for i in range(dltcoefs.shape[1]):
            output_uv[2*i:2*i+2] = dlt_inverse(dltcoefs[:,i], xyzpoint.reshape(1,-1)).flatten()
    return output_uv

def calc_paired_distance(reproj_diff):
    '''
    Parameters
    ----------
    reproj_diff : (Mpoints, Ncam*2) np.array
        Difference matrix of xy_reprojected - xy_observed.
        e.g. delta_x1, delta_y1, delta_x2, delta_y2, delta_x3, delta_y3
    Returns
    -------
    cood_dist : (Ncam,) np.array
        Euclidean distance between prediction and observed using the reprojection
        difference matrix
    '''
    camwise_coods = np.split(reproj_diff, int(reproj_diff.size/2))
    # calculate distance from the difference
    cood_dist = np.zeros(len(camwise_coods))
    for i,each in enumerate(camwise_coods):
        cood_dist[i] = np.sqrt(np.sum(each**2))
    return cood_dist



def find_best_matches(dltcoefs, cam1_tracks, cam2_tracks, cam3_tracks, max_reproj=10):
    '''
    Generates the best matching track-ids across cameras. 
    
    Parameters
    ----------
    dltcoefs : (11,Ncams) np.array
        DLT coefficients for each camera in column-wise fashion.
    cam1_tracks, cam2_tracks, cam3_tracks : pd.DataFrame
        With columns id, frame, x, y, cid
        where x,y are the object coordinates with a bottom-left origin. 
        cid is the camera id and id is the object id. 
    max_reproj : float, optional 
        Defaults to 10 pixels. The maximum tolerated euclidean distance
        between the observed and reprojected points. 

    Returns
    -------
    matching_triples : list with strings
        The point-ids separated by hyphens. For example if the pointids are
        'k1_1', 'k2_10' and 'k3_23' - one example matched id will be 
        ['k1_1-k2_10-k3_23']
    '''
    matching_triples = []
    cam2_byframe = cam2_tracks.groupby('frame')
    cam3_byframe = cam3_tracks.groupby('frame')
    for fnum, cam1_df in cam1_tracks.groupby('frame'):
        for idx, cam1_row in cam1_df.iterrows():
            point1id = cam1_row['id']
            point1_xy = cam1_row[['x','y']].to_numpy()
            try:
                cam2_frame = cam2_byframe.get_group(fnum).reset_index(drop=True)
            except KeyError:
                cam2_frame = pd.DataFrame(data=None, index=range(1),
                                          columns=cam1_tracks.columns)
                cam2_frame.loc[:,:] = np.nan

            points2_xy = cam2_frame.loc[:,['x','y']].to_numpy()
            try:
                cam3_frame = cam3_byframe.get_group(fnum).reset_index(drop=True)
            except KeyError:
                cam3_frame = pd.DataFrame(data=None, index=range(1),
                                          columns=cam1_tracks.columns)
                cam3_frame.loc[:,:] = np.nan
            points3_xy = cam3_frame.loc[:,['x','y']].to_numpy()
            all_point_combis, combi_ids = make_point_combis(point1_xy, points2_xy, points3_xy)
            all_point_combis = all_point_combis.astype(dtype=np.float64)
            all_3d_points = np.apply_along_axis(row_wise_dlt_reconstruct, 1, all_point_combis, dltcoefs)
            all_2d_reproj = np.apply_along_axis(row_wise_dlt_inverse, 1, all_3d_points, dltcoefs)
            # get 2D reprojection errors
            reprojection_errors = np.apply_along_axis(calc_paired_distance, 1, all_2d_reproj-all_point_combis)
            max_errors = np.nanmax(reprojection_errors, axis=1) # get the maximum reprojection errors across cameras
            max_errors[max_errors>=max_reproj] = np.nan
            try:
                min_error_ind = np.nanargmin(max_errors)
                cam2_candidate, cam3_candidate = combi_ids[min_error_ind]
            except:
                min_error_ind = np.nan
                cam2_candidate, cam3_candidate = np.nan,  np.nan

            try:
                best_cam2pointid = str(cam2_frame.loc[cam2_candidate, 'id'])
            except:
                best_cam2pointid = 'nan'
            try:
                best_cam3pointid = str(cam3_frame.loc[cam3_candidate, 'id'])
            except:
                best_cam3pointid = 'nan'
            matching_triples.append([point1id+'-'+best_cam2pointid+'-'+ best_cam3pointid])
    return matching_triples
