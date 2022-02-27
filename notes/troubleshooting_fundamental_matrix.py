# -*- coding: utf-8 -*-
"""The current framing of the 

@author: theja
"""
import numpy as np 
import track2trajectory.synthetic_data as syndata
import track2trajectory.projection as proj

cam1, cam2 = syndata.generate_two_synthetic_cameras_version2()


# page 246 of Hartlet & Zisserman - Table 9.1
# F = [e']_{x} P' P+
# where F is the fundamental matrix
# [e']_{x} is the epipole of the other camera - a 3x3 skew-symmetric matrix
# 
#  where e' = P'C and PC = 0

# get centre of cameras using formula on pg. 158
# C = [-M^-1 p_{4}], where M is the top-left 3x3 sub-matrix of the projection
# matrix, and p_{4]} is the last column of the projection matrix

C = np.matmul(-np.linalg.inv(cam1.cm_mtrx[:3,:3]), cam1.cm_mtrx[:,-1])
C = np.concatenate((C, np.array([1])))

C_prime = np.matmul(-np.linalg.inv(cam2.cm_mtrx[:3,:3]), cam2.cm_mtrx[:,-1])
C_prime = np.concatenate((C_prime, np.array([1])))


P = cam1.cm_mtrx
e = np.matmul(P, C_prime)



P_prime = cam2.cm_mtrx
e_prime = np.matmul(P_prime, C)


def make_skew_symmetric_matrix(a):
    '''See Appendix 4.2, page 581 of Hartley & Zisserman 2003
    for :math:`[a]_{x}`

    if a is a vector with a = (a1, a2, a3)
    then the skew-matrix of a, called [a]_{x}
    is :

        [ 0  -a3  a2
         a3   0  -a1
        -a2  a1   0]

    '''
    a1,a2,a3 = a
    a_x = np.array(([0, -a3, a2],
                    [a3, 0, -a1],
                    [-a2, a1, 0]))
    return np.float32(a_x)

e_prime_cross = make_skew_symmetric_matrix(e_prime)


# P+ is the pseudo inverse of P 
P = cam1.cm_mtrx
P_plus = np.linalg.pinv(P)

# 
Pprime_Pplus = np.matmul(P_prime, P_plus)
F = np.matmul(e_prime_cross, Pprime_Pplus)
# check that it all makes sense 
F = proj.calcFundamentalMatrix(cam1, cam2)

np.matmul(F.T,e_prime)

