'''
DLT reconstruction code
=======================
Python translation (with slight modifications) of Ty Hedrick's 
dlt_reconstruct from the easyWand package
'''
import numpy as np 

def dlt_reconstruct(c,camPts):
    '''
    TODO: 
        * Implement rmse calculation
    
    '''

    # number of cameras
    nCams = camPts.shape[1]/2
    
    #s etup output variables
    rmse = np.tile(np.nan, 3)
    
    # if we have 2+ cameras, begin reconstructing
    if nCams < 2:
        raise ValueError(f'At least >= 2 camera xy points must be there. nCams:{nCams}')

    all_camera_m1 = []
    all_camera_m2 = []
    for i in range(int(nCams)):
        u,v = camPts[:,2*i], camPts[:,2*i+1]
        m1_thiscam = np.zeros((2,3))
        m2_thiscam = np.zeros((2,1))
        
        m1_thiscam[0,0] = u*c[8,i] - c[0,i]
        m1_thiscam[0,1] = u*c[9,i] - c[1,i]
        m1_thiscam[0,2] = u*c[10,i] - c[2,i]
        m1_thiscam[1,0] = v*c[8,i] - c[4,i]
        m1_thiscam[1,1] = v*c[9,i] - c[5,i]
        m1_thiscam[1,2] = v*c[10,i] - c[6,i]

        all_camera_m1.append(m1_thiscam)
        
        m2_thiscam[0] = c[3,i]-u
        m2_thiscam[1] = c[7,i]-v
        all_camera_m2.append(m2_thiscam)
   
    m1_overall = np.row_stack(all_camera_m1)
    m2_overall = np.row_stack(all_camera_m2)
        
    xyz = np.linalg.lstsq(m1_overall, m2_overall)
    return  xyz 
