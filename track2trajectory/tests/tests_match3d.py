'''
Tests for match2dto3d
=====================

'''
import numpy as np 
import pandas as pd
import unittest
import track2trajectory.match3d as m3d
import track2trajectory.match2d as m2d
import track2trajectory.synthetic_data as syndata
import track2trajectory.projection as projection
#np.random.seed(78462)

class TestEstimate3DPoints(unittest.TestCase):
    '''Does it even run
    '''
    def setUp(self):
        '''Generate 2 synthetic cameras with multiple objects in the field of view
        '''
        self.cam1, self.cam2 = syndata.generate_two_synthetic_cameras_version2()
        x_range = np.linspace(-0.5,0.5,2)
        y_range = np.linspace(5,10,2)
        z_range = np.linspace(-1,1,2)
        
        threed_data = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)
        self.points_in_3d = pd.DataFrame(data={'x':threed_data[:,0],
                                               'y':threed_data[:,1],
                                               'z':threed_data[:,2],
                                               'frame':np.tile(1, threed_data.shape[0]),
                                               'oid':np.arange(threed_data.shape[0])})

        self.all_2dpoints = self.make_2dprojection_data(self.points_in_3d)
        self.fundamatrix_cam2fromcam1 = projection.calcFundamentalMatrix(self.cam1, self.cam2)
        
    def make_2dprojection_data(self, points_in_3d):
        
        cam1_2dpoints, cam1_fails = projection.project_to_2d_and_3d(points_in_3d,
                                                                         self.cam1)
        cam2_2dpoints, cam2_fails  = projection.project_to_2d_and_3d(points_in_3d,
                                                                          self.cam2)
        all_2dpoints = pd.concat([cam1_2dpoints,
                                  cam2_2dpoints]).reset_index(drop=True)
        return all_2dpoints

    def test_doesEstimate3DPointsRun(self):
        '''
        No Kalman filtering yet here
        '''
        
        print('...........miaow')
        output = m3d.estimate_3d_points(self.cam1, self.cam2, 
                                 self.all_2dpoints, self.points_in_3d,
                                 'output_testresuts', self.fundamatrix_cam2fromcam1,
                                 do_kalman_filter_predictions=False)
        print(output)
        

if __name__ == '__main__':
    unittest.main()