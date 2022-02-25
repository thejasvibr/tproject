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
from track2trajectory import make_homog


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
        self.cam1, self.cam2 = syndata.generate_two_synthetic_cameras_version2()
        self.F = projection.calcFundamentalMatrix(self.cam1, self.cam2)

        self.C = np.matmul(-np.linalg.inv(self.cam1.cm_mtrx[:3,:3]),
                                                   self.cam1.cm_mtrx[:,-1])
        self.C = np.concatenate((self.C, np.array([1])))
        self.P_prime = self.cam2.cm_mtrx
        self.e_prime = np.matmul(self.P_prime, self.C)
        
        self.Cprime = np.matmul(-np.linalg.inv(self.cam2.cm_mtrx[:3,:3]), 
                           self.cam2.cm_mtrx[:,-1])
        self.Cprime = np.concatenate((self.Cprime, np.array([1])))
        self.P = self.cam1.cm_mtrx
        self.e = np.matmul(self.P, self.Cprime)

    def test_PC_relations(self):
        PCisZero = np.all(np.matmul(self.P, self.C)==0)
        primePCisZero = np.all(np.matmul(self.P_prime, self.Cprime)==0)
        self.assertTrue(np.all([PCisZero, primePCisZero]))

    def test_epipole_relations(self):
        '''
        Fe = 0
        
        and 
        
        F.T e' = 0
        '''
        epipole_condition1 = np.matmul(self.F, self.e)
        condition1_satisfied = np.allclose(epipole_condition1, [0,0,0], atol=1e-6)
        epipole_condition2 = np.matmul(self.F.T, self.e_prime)
        condition2_satisfied = np.allclose(epipole_condition2, [0,0,0], atol=1e-6)
        self.assertTrue(np.all([condition1_satisfied, condition2_satisfied]))


    def test_xprime_Fx_relation(self):
        '''
        '''
        threed_data = np.array(([0,10,0],
                              [-1,9,-1])).reshape(-1,3)
        points_in_3d = pd.DataFrame(data={'x':threed_data[:,0],
                                               'y':threed_data[:,1],
                                               'z':threed_data[:,2],
                                               'frame':np.tile(1, threed_data.shape[0]),
                                               'oid':np.arange(threed_data.shape[0])})

        cam1_2dpoints, _ = projection.project_to_2d_and_3d(points_in_3d,
                                                               self.cam1)
        cam2_2dpoints, _ = projection.project_to_2d_and_3d(points_in_3d,
                                                               self.cam2)

        xy_cam1 = cam1_2dpoints.loc[0,['x','y']].to_numpy()
        X = make_homog(xy_cam1)
        xy_cam2 = cam2_2dpoints.loc[0,['x','y']].to_numpy()
        Xprime = make_homog(xy_cam2)

        xprime_F_x = np.matmul(np.matmul(Xprime.T,self.F), X)
        xprimeFx_closetozero = abs(xprime_F_x) < 1e-9
        self.assertTrue(xprimeFx_closetozero)

if __name__ == '__main__':
    unittest.main()