"""
Gravity alignment code
======================

@author: theja
"""
import matplotlib.pyplot as plt 
from scipy.spatial.transform import Rotation
import numpy as np 
np.random.seed(82319)


t = np.linspace(0,3,75)
xyz = np.zeros((t.size,3))
h = 1.5
v = 10
theta = np.pi/3
g = 9.8
xyz[:,2] = h + v*t*np.sin(theta) - (g*t**2)/2
xyz[:,1] =  v*t*np.cos(theta)
# add noise to the estimates 
xyz += np.random.normal(0,0.2,xyz.size).reshape(xyz.shape)
# also rotate all the coordinates a bit

def rotate_point(point):
    rotmat = Rotation.from_euler('zxy', [10,-15,-70],degrees=True).as_matrix()
    return np.matmul(rotmat, point)

xyz_rot = np.apply_along_axis(rotate_point, 1, xyz)


#%%

fps = 25
def smooth_2nddeg(xyz):
    t = np.arange(xyz.shape[0])
    xyz_fit = [np.polyfit(t,xyz[:,each], 2) for each in range(xyz.shape[1])]
    # now calculate the smoothed positions
    xyz_smooth = [np.polyval(each, t) for each in xyz_fit]
    xyz_sm = np.column_stack([each for each in xyz_smooth])
    return xyz_sm

def calc_acc(xyz_smooth, fps):
    v = np.apply_along_axis(np.diff, 0, xyz_smooth)
    acc = np.apply_along_axis(np.diff, 0, v)/(1.0/(fps**2))
    return acc

def smooth_and_acc(xyz, fps=25):
    smooth_xyz = smooth_2nddeg(xyz)
    acc = calc_acc(smooth_xyz, fps)
    return acc

def calc_norm(X):
    return np.linalg.norm(X)

def row_calc_norm(X):
    return np.apply_along_axis(calc_norm, 1, X)
    


own_sm = smooth_and_acc(xyz_rot)
orig_sm = smooth_and_acc(xyz)

mean_acc = np.mean(own_sm,0)
mean_acc/np.linalg.norm(mean_acc)

# if __name__ == '__main__':
#     plt.figure()
#     plt.plot(xyz_rot)

