# -*- coding: utf-8 -*-
"""
Helper functions
================
"""
import numpy as np

def make_homog(X):
    '''
    Appends a 1 onto X, and converts it from a 
    (n,) to a (n+1,) np.array
    '''
    homog_X = np.concatenate((X.flatten(), np.array([1]))).flatten()
    return homog_X


def get_closest_points(points2d, distances, **kwargs):
    '''
    Parameters
    ----------
    points2d : (N,2) np.array
        xy points 
    distances : list-like
        Distances of the 2D points from a given epipolar line
    rel_threshold : float >0 , optional
        Defaults to 0.2

    Returns
    -------
    close_points : (N,2) np.array
        Points that are around the same low distance from the epipolar line
    '''
    rel_threshold = kwargs.get('rel_threshold', 0.2)
    distances = np.array(distances)
    if not np.logical_and(rel_threshold > 0, rel_threshold<=1):
        raise ValueError(f'Relative threshold must be >0 and <= 1: {rel_threshold}')
    if np.nanmin(distances) ==0:
        small_distances = np.argwhere(distances==0).flatten()
    else:
        normalised_distances = distances/np.nanmin(distances)
        # distances considered close
        small_distances = np.argwhere(normalised_distances <= 1+rel_threshold).flatten()
    # print(f'points with low distance: {small_distances}')
    close_points = points2d[small_distances,:]
    return close_points

def check_unique_oids_per_frame(twod_points):
    '''
    Checks that there are no repeated object IDs in each of the frames.
    '''
    pass