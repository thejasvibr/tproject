# -*- coding: utf-8 -*-
"""
Prediction tests
================
"""
import numpy as np 
import pandas as pd
import scipy.spatial as spatial
import unittest

from track2trajectory import kalman_predict

class SimpleKalmanTests(unittest.TestCase):
    '''
    Check the on-axis levels over different cap apertures and ka values
    '''
    
    def setUp(self):
        '''
        Generates a gentle 3D curve with very low frequency
        '''
        frames = 20
        t_frames = np.linspace(0,0.5,frames)
        x, y, z = np.sin(2*np.pi*t_frames*0.2), np.cos(2*np.pi*t_frames*0.1), np.zeros(frames)
        # frame, objectid_cam1, objectid_cam2, x, y, z
        self.df = pd.DataFrame(data={'x':x,'y':y,'z':z,'objectid_cam1':np.tile(1, frames),
                                'frame':np.arange(frames)})
        self.num_frames_used = 12
        self.part_df = self.df.loc[:self.num_frames_used,:]
        
    def test_3d_sinecurve(self):
        '''
        Just check that the Kalman Filter produces an output that is within 10 cm of 
        expected
        '''
        predictions = kalman_predict(1, self.num_frames_used+1, self.part_df)
        xyz = predictions[:-1]
        num_frames = predictions[-1]
        euclidean_error = spatial.distance.euclidean(xyz, self.df.loc[self.num_frames_used+1,['x','y','z']])
        self.assertTrue(euclidean_error<0.1)
        

class ForwardBackwardPassTest(unittest.TestCase):
    '''
    '''
    
    def test_backwardforward(self):
        raise NotImplementedError('double direction pass not tested yet')

if __name__=='__main__':
    unittest.main()


