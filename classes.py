import numpy as np


class Camera:
    def __init__(self, c_id, pos, f, c_x, c_y, f_x, f_y, i_mtrx, t_mtrx, r_mtrx, cof_mtrx, rel_pos, cm_mtrx):
        self.id = c_id
        self.pos = np.float32(pos)
        self.f = np.float32(f)
        self.c_x = np.float32(c_x)
        self.c_y = np.float32(c_y)
        self.f_x = np.float32(f_x)
        self.f_y = np.float32(f_y)
        self.i_mtrx = np.float32(i_mtrx)
        self.t_mtrx = np.float32(t_mtrx)
        self.r_mtrx = np.float32(r_mtrx)
        self.cof_mtrx = np.float32(cof_mtrx)
        self.fmm = np.float32(f * 0.00576 / 1920.0)
        self.rel_pos = np.float32(rel_pos)
        self.cm_mtrx = np.float32(cm_mtrx)