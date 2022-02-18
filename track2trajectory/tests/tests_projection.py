# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 18:58:32 2022

@author: theja
"""
import numpy as np 
import pandas as pd
import unittest 
import track2trajectory.projection as projection
import track2trajectory.synthetic_data as syndata
from scipy.spatial.transform import Rotation as R



class Check2DProjectionWorks(unittest.TestCase):
    
    def setUp(self):
        '''
        '''
        cameras123 = syndata.generate_three_synthetic_cameras()
        self.camera1 = cameras123[0]
        self.camera2 = cameras123[1]
        self.camera3 = cameras123[2]
        
        # generate mock 3D data
        num_frames = 20
        data_xyz= {'frame' : np.arange(num_frames),
               'x' : np.random.normal(-1, 0.1, num_frames),
               'y' : np.random.normal(-1, 0.1, num_frames),
               'z' : np.random.normal(1, 0.1, num_frames)}
        self.point_xyz = pd.DataFrame(data=data_xyz)
        self.point_xyz['oid'] = 1

    def test_check2Dprojectionworks(self):
        twod_projections, num_failed = projection.project_to_2d_and_3d(self.point_xyz, 
                                                        self.camera1,
                                                        )
        print(num_failed)

class Verify2DProjectionResults(unittest.TestCase):
    '''
    Sets up one simulated camera at -1,0,0 facing towards the origin.
    Test points are placed at +ve X and 0 Y coordinates, and the projection is 
    checked.
    '''
    def setUp(self):
        # rotate cam1 to the origin, a -90 degree rotation about its y axis
        rotation_to_origin = R.from_euler('xyz', [0,-90,0],degrees=True).as_matrix()
        args = {'cam1_Rotation':rotation_to_origin}
        self.cam1, self.cam2 = syndata.generate_two_synthetic_cameras_version2(**args)
        self.points_in_space = pd.DataFrame(data={'frame':[1,2,3],
                                             'x':[1,2,2],
                                             'y':[0,0,0],
                                             'z':[0,0.25,0.5],
                                             'oid': [1]*3,
                                             'cid':[1]*3})

    def test_2dprojection(self):
        projected_points, _ = projection.project_to_2d_and_3d(self.points_in_space,
                                                              self.cam1)

        self.assertTrue(len(projected_points)>0)
        # check that the pixel coordinates are always the principal x point.
        ppx_matches = np.array_equal(np.int64(projected_points['x']), np.tile(960,3))
        self.assertTrue(ppx_matches)

class CheckFundamentalMatrixCalculationWorks(unittest.TestCase):
    
    def setUp(self):
        '''
        '''
        cameras123 = syndata.generate_three_synthetic_cameras()
        self.camera1 = cameras123[0]
        self.camera2 = cameras123[1]
        self.camera3 = cameras123[2]
    def test_fundamentalmatrixworks(self):
        fundamental_matrix = projection.calcFundamentalMatrix(self.camera1,
                                                              self.camera2)


class TestFundamentalMatrixCalculation(unittest.TestCase):
    '''Check if the output fundamental matrix makes mathematical sense
    The expected relations satisfied by the fundamental matrix are on page 246 of 
    Hartley & Zisserman 2003
    '''
    def setUp(self):
        cam1, cam2 = syndata.generate_two_synthetic_cameras_version2()
        self.F = projection.calcFundamentalMatrix(cam1, cam2)
        
            
        C = np.matmul(-np.linalg.inv(cam1.cm_mtrx[:3,:3]), cam1.cm_mtrx[:,-1])
        C = np.concatenate((C, np.array([1])))
        P_prime = cam2.cm_mtrx
        self.e_prime = np.matmul(P_prime, C)
        
        Cprime = np.matmul(-np.linalg.inv(cam2.cm_mtrx[:3,:3]), cam2.cm_mtrx[:,-1])
        Cprime = np.concatenate((Cprime, np.array([1])))
        P = cam1.cm_mtrx
        self.e = np.matmul(P, Cprime)

    def test_epipole_relations(self):
        epipole_condition1 = np.matmul(self.F.T, self.e)==0
        epipole_condition2 = np.matmul(self.F.T, self.e_prime)==0
        self.assertTrue(np.all([epipole_condition1, epipole_condition2]))


if __name__ == '__main__':
    unittest.main()