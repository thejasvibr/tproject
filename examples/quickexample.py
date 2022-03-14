# -*- coding: utf-8 -*-
"""
Quick run through example
=========================
This example will show the basic commands required to run the 2d trajectory 
matching and 3D position triangulation.

Author: Thejasvi Beleyur, March 2022
"""
import uuid
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from track2trajectory.synthetic_data import generate_two_synthetic_cameras_version2
from track2trajectory.synthetic_data import make_brownian_particles
from track2trajectory.projection import project_to_2d_and_3d, calcFundamentalMatrix
from track2trajectory.match3d import match_2dpoints_to_3dtrajectories
#%%
# Generating synthetic data
# -------------------------
# Let's first generate 1) 2 cameras located at `xyz` [-1,0,0] and [1,0,0].
# and 2) calculate the fundmental matrix mapping points between the two cameras.
cam1, cam2 = generate_two_synthetic_cameras_version2()
F = calcFundamentalMatrix(cam1, cam2)

#%% Now let's proceed to synthesize a few particles moving around in front of the 
# two cameras - such that the particles stay within the common fields of view
# of both cameras most of the time. 
num_particles  = 10
stepsize = 0.1 # in m
bounding_box = [[-4,4],
                [7,10],
                [-1,1]] # set the XYZ min-max limits for the bounding box
xyz_particles = make_brownian_particles(num_particles, bounding_box)
#%% Project the 3d particle positions onto the camera sensors
cam1_2dpoints, cam1_fails = project_to_2d_and_3d(xyz_particles, cam1)
cam2_2dpoints, cam2_fails = project_to_2d_and_3d(xyz_particles, cam2)
# Change the object IDs on cam2 - just to show that the object IDs can be 
# any type alphanumeric code. 

new_codes = [str(uuid.uuid4())[-4:] for each in range(num_particles)]
replacement = { each:new_codes[i]  for i,each in enumerate(cam2_2dpoints['oid'].unique())}
cam2_2dpoints['oid'] = cam2_2dpoints['oid'].replace(replacement)

cam2_2dpoints
#%% 
# Running trajectory matching
# ---------------------------
# Let's now run the :code:`match_2dpoints_to_3dtrajectories` function, which
# checks for correspondences between points on each camera. If a correspondence
# is found, then a 3D point is calculated from it.

threed_matches = match_2dpoints_to_3dtrajectories(cam1, cam2, 
                                                  cam1_2dpoints,
                                                  cam2_2dpoints, F)

threed_matches

