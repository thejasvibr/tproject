# -*- coding: utf-8 -*-
"""
Get camera world centre and orientation from DLT
================================================
Based on Ty Hedrick's MATLAB script hosted at https://biomech.web.unc.edu/dlt-to-from-intrinsic-extrinsic/
"""
import numpy as np 
import pandas as pd

def cam_centre_from_dlt(coefs):
    '''
    
    
    Reference
    ---------
    * http://www.kwon3d.com/theory/dlt/dlt.html Equation 25
    '''
        
    m1 = np.array([[coefs[0],coefs[1],coefs[2]],
                 [coefs[4],coefs[5],coefs[6]],
                 [coefs[8],coefs[9],coefs[10]]])
    m2 = np.array([-coefs[3], -coefs[7], -1]).T

    xyz = np.matmul(np.linalg.inv(m1),m2)
    return xyz

def transformation_matrix_from_dlt(coefs):
    '''
    Based on the DLTcameraPosition.m function written by Ty Hedrick. 
    
    Parameters
    ----------
    coefs : (11,Ncamera) np.array
    
    Returns
    -------
    T : (4,4) np.array
        Transformation matrix
    Z : float
        Distance of camera centre behind image plane
    ypr : (3,) np.array
        yaw, pitch, roll angles in degrees
    
    
    Notes
    -----
    I guess this function is based on the equations described in 
    Kwon3d (http://www.kwon3d.com/theory/dlt/dlt.html#3d).
            
    The transformation matrix T -
    
    
    
    ''' 
    D = (1/(coefs[8]**2+coefs[9]**2+coefs[10]**2))**0.5;
    #D = D[0]; # + solution
    
    Uo=(D**2)*(coefs[0]*coefs[8]+coefs[1]*coefs[9]+coefs[2]*coefs[10]);
    Vo=(D**2)*(coefs[4]*coefs[8]+coefs[5]*coefs[9]+coefs[6]*coefs[10]);
    print(f'D: {D}, Uo: {Uo}, Vo:{Vo}')
    du = (((Uo*coefs[8]-coefs[0])**2 + (Uo*coefs[9]-coefs[1])**2 + (Uo*coefs[10]-coefs[2])**2)*D**2)**0.5;
    dv = (((Vo*coefs[8]-coefs[4])**2 + (Vo*coefs[9]-coefs[5])**2 + (Vo*coefs[10]-coefs[6])**2)*D**2)**0.5;
    
    #du = du[0]; # + values
    #dv = dv[0]; 
    Z = -1*np.mean([du,dv]) # there should be only a tiny difference between du & dv
    
    row1 = [(Uo*coefs[8]-coefs[0])/du ,(Uo*coefs[9]-coefs[1])/du ,(Uo*coefs[10]-coefs[2])/du]
    row2 = [(Vo*coefs[8]-coefs[4])/dv ,(Vo*coefs[9]-coefs[5])/dv ,(Vo*coefs[10]-coefs[6])/dv] 
    row3 = [coefs[8] , coefs[9], coefs[10]]
    T3 = D*np.array([row1,
                     row2,
                     row3])

    dT3 = np.linalg.det(T3);
    
    if dT3 < 0:
      T3=-1*T3;
    
    xyz = cam_centre_from_dlt(coefs)
    
    T = np.zeros((4,4))
    T[:3,:3] = np.linalg.inv(T3);
    T[3,:]= [xyz[0], xyz[1], xyz[2], 1]
    
        
    # % compute YPR from T3
    # %
    # % Note that the axes of the DLT based transformation matrix are rarely
    # % orthogonal, so these angles are only an approximation of the correct
    # % transformation matrix
    # %  - Addendum: the nonlinear constraint used in mdlt_computeCoefficients below ensures the
    # %  orthogonality of the transformation matrix axes, so no problem here
    alpha = np.arctan2(T[1,0],T[0,0])
    beta = np.arctan2(-T[2,0], (T[2,1]**2+T[2,2]**2)**0.5)
    gamma = np.arctan2(T[2,1],T[2,2])
    
    ypr = np.rad2deg([gamma,beta,alpha]);

    return T, Z, ypr

   
def assign_coefs_rowwise(coefs):
    dlt_mat = np.zeros((3,4))
    k = 0 
    for i in range(dlt_mat.shape[0]):
        for j in range(dlt_mat.shape[1]):
            if not k>10:
                dlt_mat[i,j] = coefs[k]
                k+=1
            else:
                pass
    dlt_mat[-1,-1] = 0
    return dlt_mat