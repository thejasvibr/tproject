# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 13:18:00 2022

@author: theja
"""
import unittest
import string
import numpy as np 
import pandas as pd
from track2trajectory import projection
from track2trajectory import synthetic_data as syndata
import track2trajectory.match2d as m2d
from track2trajectory.projection import project_to_2d_and_3d, calcFundamentalMatrix
from track2trajectory.match2d import generate_2d_correspondences
np.random.seed(82319)

class TestFindCandidateWorks(unittest.TestCase):

    def setUp(self):
        '''
        Generate mock data of objects in a 3D grid. This might lead to
        multiple 2D points corresoponding to each other - but that's fine too.
        '''
        self.cam1, self.cam2 = syndata.generate_two_synthetic_cameras_version2()
        x_range = np.tile(0,3)
        y_range = np.linspace(10,20,3)
        z_range = np.linspace(-1,1,4)
        all_points = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)
        self.points_in_3d = pd.DataFrame(data={'x':all_points[:,0],
                                               'y':all_points[:,1],
                                               'z':all_points[:,2],
                                               'frame':np.tile(1, all_points.shape[0]),
                                          'oid':np.arange(all_points.shape[0])})

        self.fundamental_matrix = projection.calcFundamentalMatrix(self.cam1, self.cam2)
        self.cam1_2dpoints, cam1_fails = projection.project_to_2d_and_3d(self.points_in_3d,
                                                                         self.cam1)
        self.cam2_2dpoints, cam2_fails  = projection.project_to_2d_and_3d(self.points_in_3d,
                                                                          self.cam2)

        self.fundamatrix_cam2fromcam1 = projection.calcFundamentalMatrix(self.cam1, self.cam2)
        # print(f'Cam1 2d points: {self.cam1_2dpoints}')
        # print(f'Cam2 2d points: {self.cam2_2dpoints}')

    def test_runfindcandidate(self):

        for i, (frame, oid, x,y, cid) in self.cam1_2dpoints.iterrows():
            # print(f'i ={i}')
            candidate_point = np.float32([x,y])

            # check that the candidate is always the correct corresponding point.
            matching_output = m2d.find_candidates(self.cam2_2dpoints,
                                                   self.fundamatrix_cam2fromcam1,
                                                   self.cam2,
                                                   self.cam1,candidate_point,
                                                   )
            matched_point = np.float32(matching_output[0])
            valid_matched_point = np.invert(np.all(np.isnan(matched_point))) and matched_point.size>1
            #if matched_point
            valid_candidate = np.all(np.invert(np.isnan(candidate_point)))

            expected_matchedpoint = np.float32(self.cam2_2dpoints.loc[i,['x','y']])
            valid_corresponding_cam2point =  np.all(np.invert(np.isnan(expected_matchedpoint)))
            if valid_matched_point and valid_candidate and valid_corresponding_cam2point:
                points_equal = np.allclose(matched_point, 
                                           expected_matchedpoint,atol=1e-4)
                self.assertTrue(points_equal)

    def test_nancase_findcandidate(self):
        '''What happens when there are np.nan x and y coordinates
        '''

        # check that the candidate is always the correct corresponding point.
        nan_output = m2d.find_candidates(self.cam2_2dpoints,
                                               self.fundamatrix_cam2fromcam1,
                                               self.cam1,
                                               self.cam2,np.float32([np.nan, np.nan]),
                                               )
        output_match = np.sum(np.isnan(nan_output))==2
        self.assertTrue(output_match)

class CheckGenerate2DCorrespondences(unittest.TestCase):

    def setUp(self):
        self.cam1, self.cam2 = syndata.generate_two_synthetic_cameras_version2()
        num_particles = 15
        num_frames = 20
        points_in_3d = syndata.make_brownian_particles(num_particles,
                                                        [[-5,5],
                                                            [7,10],
                                                            [-1,1]], frames=num_frames,
                                                        stepsize=0.2)

        self.all_2dpoints = self.make_2dprojection_data(self.cam1, self.cam2, points_in_3d)
        self.F = calcFundamentalMatrix(self.cam1, self.cam2)
        self.cam1_2dpoints = self.all_2dpoints[self.all_2dpoints['cid']==0].reset_index(drop=True)
        self.cam2_2dpoints = self.all_2dpoints[self.all_2dpoints['cid']==1].reset_index(drop=True)
        # alter the camera 2 oids just to check that it's not specific to a numerical oid
        oid_replacement_dict = {each: string.ascii_lowercase[each]  for each in range(num_particles)}
        self.cam2_2dpoints['oid'] = self.cam2_2dpoints['oid'].replace(oid_replacement_dict)
        
    def make_2dprojection_data(self, cam1, cam2, points_in_3d):
        
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

    def test_2dCorrespondenceRuns(self):
        correspondences, fails = generate_2d_correspondences(self.cam1,
                                                              self.cam2,
                                                              self.cam1_2dpoints,
                                                              self.cam2_2dpoints,
                                                              self.F)
    def test_remove_1_frame_data(self):
        '''Mimic the situation when we have no data on one frame in one camer. 
        Here, let's remove the data from cam2 Frame 2
        '''
        cam2_woframe2 = self.cam2_2dpoints.copy()
        cam2_woframe2 = cam2_woframe2[cam2_woframe2['frame']!=2].reset_index(drop=True)
        correspondences, fails = generate_2d_correspondences(self.cam1,
                                                              self.cam2,
                                                              self.cam1_2dpoints,
                                                              cam2_woframe2,
                                                              self.F)


if __name__ == '__main__':
    unittest.main()