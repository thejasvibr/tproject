'''
Camera
=======
Contains the Camera class
'''

import numpy as np


class Camera:
    '''
    Generates camera objects with all specified parameters

    
    '''
    def __init__(self, c_id, pos, f, c_x, c_y, f_x, f_y,
                         i_mtrx, t_mtrx, r_mtrx, cof_mtrx, rel_pos, cm_mtrx):
        '''

        Parameters
        ----------
        c_id : int
            camera id.
        pos : (1,1,2) np.array
            ? Epipolar point ? (the intersection point on image plane
            of the line between two camera centers) ?? 
            Doesn't seem to be used in the code - can possibly be removed
        f : float
            focal length in pixels
        c_x : float
            camera x centre in pixels
        c_y : float
            camera y centre in pixels
        f_x : float
            DESCRIPTION.
        f_y : TYPE
            DESCRIPTION.
        i_mtrx : 3x3 np.array
            Intrinsic matrix of the form 
            f 0 c_x  
            0 f c_y 
            0 0  1 
        t_mtrx : (3,) or (3,1) np.array
            translation matrix
        r_mtrx : 3x3  np.array
            Rotation matrix
        cof_mtrx : (N,) np.array
            Distortion coefficients 
        rel_pos : TYPE
            DESCRIPTION.
        cm_mtrx : 3x4 np.array
            Projection matrix, this is the intrinsic matrix X R matrix
            where the R matrix is the 3x4 matrix made of the 3x3 rotation 
            matrix and the translation (3x1) matrix 

        '''
        self.id = c_id # camera ID
        self.pos = np.float32(pos) # ???
        self.f = np.float32(f) # 
        self.c_x = np.float32(c_x) 
        self.c_y = np.float32(c_y)
        self.f_x = np.float32(f_x)
        self.f_y = np.float32(f_y)
        self.i_mtrx = np.float32(i_mtrx) # intrinsic matrix
        self.t_mtrx = np.float32(t_mtrx) # translation matrix
        self.r_mtrx = np.float32(r_mtrx) # rotation matrix
        self.cof_mtrx = np.float32(cof_mtrx) # distortion coefficient matrix
        self.fmm = np.float32(f * 0.00576 / 1920.0) # focal length in mm (is calculated independently if not provided)
        self.rel_pos = np.float32(rel_pos)
        self.cm_mtrx = np.float64(cm_mtrx) # camera matrix