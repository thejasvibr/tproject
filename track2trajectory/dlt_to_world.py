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
    Get the transformation from camera 3d to world 3d.

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


def dlt_inverse(c, xyz):
    '''
    A Python port of the dlt_inverse MATLAB function by Ty Hedrick from 
    DLTdv7
    
    Generates 2D pixels of a 3D point given the 3D xyz coordinates and 
    camera DLT coefficients. 
    
    Parameters
    ----------
    c : (11,) np.array
        11 DLT coefficients
    xyz : (nframes, 3) np.array
        N x 3d xyz coordinates 
    
    Returns
    -------
    uv : (nframes,2) np.array
        row and col pixels across frames
    '''
    nrows = xyz.shape[0]
    
    uv = np.empty((nrows,2))
    uv[:,:] = np.nan
    uv_col0_numerator = xyz[:,0]*c[0] + xyz[:,1]*c[1] + xyz[:,2]*c[2] + c[3]
    uv_col0_denom = xyz[:,0]*c[8]+xyz[:,1]*c[9]+xyz[:,2]*c[10]+1
    uv[:,0] = uv_col0_numerator/uv_col0_denom
    
    uv_col1_num = xyz[:,0]*c[4]+xyz[:,1]*c[5]+xyz[:,2]*c[6]+c[7]
    uv_col1_denom = xyz[:,0]*c[8]+xyz[:,1]*c[9]+xyz[:,2]*c[10]+1
    uv[:,1] = uv_col1_num/uv_col1_denom
    return uv
    

def partialdlt(u,v,C1, C2):
    '''
    A Python port of the partialdlt MATLAB function by Ty Hedrick from DLTdv7
    
    Parameters
    ----------
    u, v: floats
        x,y coods of a point in cam 1
    C1, C2 : (11,) np.array
        Array with 11 DLT coefficients 
    Returns
    -------
    m : float
        Slope of the epipolar line 
    b : float
        Y -intercept of the epipolar line on cam2 

    '''
        
   
    # pick 2 random Z (actual values are not important)
    z = np.array([500, -500])
    
    # for each z predict x & y
    x = np.tile(np.nan, 2);
    y = np.tile(np.nan, 2);
    for i in range(2):
      Z = z[i]
      pt1 = u*C1[8]*C1[6]*Z + u*C1[8]*C1[7] - u*C1[10]*Z*C1[4] -u*C1[4]
      pt2 = C1[0]*v*C1[10]*Z + C1[0]*v - C1[0]*C1[6]*Z - C1[0]*C1[7]
      pt3 = -C1[2]*Z*v*C1[8] + C1[2]*Z*C1[4] - C1[3]*v*C1[8] + C1[3]*C1[4]
      denom =   u*C1[8]*C1[5] - u*C1[9]*C1[4] + C1[0]*v*C1[9] - C1[0]*C1[5] -C1[1]*v*C1[8] + C1[1]*C1[4]
      y[i]= (pt1 + pt2 + pt3 )/denom
      y[i] *= -1

      Y = y[i]
      
      x[i]= (v*C1[9]*Y+v*C1[10]*Z+v-C1[5]*Y-C1[6]*Z-C1[7])/(v*C1[8]-C1[4])
      x[i] *= -1 
    
    # back the points into the cam2 X,Y domain
    xy = np.zeros((2,2));
    for i in range(2):
      xy[i,:]=dlt_inverse(C2,np.array([x[i],y[i],z[i]]).reshape(-1,3))
    
    # get a line equation back, y=mx+b
    m = (xy[1,1]-xy[0,1])/(xy[1,0]-xy[0,0]);
    b = xy[0,1] - m*xy[0,0];
    return m, b
    




