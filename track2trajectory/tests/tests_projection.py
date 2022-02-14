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


        



if __name__ == '__main__':
    unittest.main()