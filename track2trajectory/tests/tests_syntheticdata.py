'''
Synthetic data tests
====================
Runs tests on functions which generate synthetic cameras, 3D and 2D points
'''

import pandas as pd
import unittest 
import track2trajectory.synthetic_data as syndata


class Check3SynCamWorks(unittest.TestCase):
    
    def test_workingcondition(self):
        syndata.generate_three_synthetic_cameras()

class CheckCreateCameraWorks(unittest.TestCase):
    def setUp(self):
        '''
        Make the dataframe which has all required parameters for create_camera
        '''
        cameras123 = syndata.generate_three_synthetic_cameras()
        camera1 = cameras123[0]
        cam1_data = {'focal_length':[camera1.f], 'ppx':[960],'ppy':[540],
                     'x':[0], 'y':[0], 'z':[2]}
        for each in ['coe3','coe4','coe5']:
            cam1_data[each] = 0 
        # some representative o1,2,3,4 from the starling dataset camera xcp file
        for variable, value in zip(['o1','o2','o3','o4','ppx','ppy'], [0.85, 0.085, -0.06, 0.50, 960, 540]):
            cam1_data[variable] = [value]
        
        self.camera_df = pd.DataFrame(data=cam1_data)
        self.camera_df['did'] = 10
        # ,'o1','o2','o3','o4'
    def test_createcamworks(self):
        created_camera = syndata.create_camera(self.camera_df, 0)

if __name__ == '__main__':
    unittest.main()