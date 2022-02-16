'''
Tests for match2dto3d
=====================

'''
import numpy as np 
import pandas as pd
import unittest
import track2trajectory.match2dto3d as m2d3d
import track2trajectory.synthetic_data as syndata
import track2trajectory.projection as projection
#np.random.seed(78462)

class TestFindCandidateWorks(unittest.TestCase):

    def setUp(self):
        self.cam1, self.cam2 = syndata.generate_two_synthetic_cameras_version2()

        x,y,z = np.random.normal(0,1,10), np.random.normal(2,1,10), np.random.normal(0,1,10)
        self.points_in_3d = pd.DataFrame(data={'x':x,'y':y,'z':z,'frame':np.tile(1, x.size),
                                          'oid':np.arange(10)})

        self.fundamental_matrix = projection.calcFundamentalMatrix(self.cam1, self.cam2)
        self.cam1_2dpoints, cam1_fails = projection.project_to_2d_and_3d(self.points_in_3d,
                                                                         self.cam1)
        self.cam2_2dpoints, cam2_fails  = projection.project_to_2d_and_3d(self.points_in_3d,
                                                                          self.cam2)
        self.fundamatrix_cam2fromcam1 = projection.calcFundamentalMatrix(self.cam1, self.cam2)

    def test_runfindcandidate(self):
        for i, (frame, oid, x,y, cid) in self.cam1_2dpoints.iterrows():
            candidate_point = np.float32([x,y])
            # check that the candidate is always the correct corresponding point.
            matching_output = m2d3d.find_candidate(self.cam2_2dpoints,
                                                   self.fundamatrix_cam2fromcam1,
                                                   self.cam1,
                                                   self.cam2,candidate_point,
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
        matching_output = m2d3d.find_candidate(self.cam2_2dpoints,
                                               self.fundamatrix_cam2fromcam1,
                                               self.cam1,
                                               self.cam2,np.float32([np.nan, np.nan]),
                                               )
        self.assertEqual(matching_output[0],[])

if __name__ == '__main__':
    unittest.main()