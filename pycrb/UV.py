import numpy as np


class UV(object):
    """(u,v)-space class"""

    def __init__(self, u_, v_):

        self.u_res = u_
        self.v_res = v_
        self.u = np.linspace(-1, 1, self.u_res)
        self.v = np.linspace(1, -1, self.v_res)

    def steer(self, u_s, v_s):

        self.u = np.repeat(u_s, self.u_res)
        self.v = np.repeat(v_s, self.v_res)

    def desteer(self):

        self.u = np.linspace(-1, 1, self.u_res)
        self.v = np.linspace(-1, 1, self.u_res)

    def _visible_region(self):

        U_, V_ = np.meshgrid(self.u, self.v)

        Z = U_**2 + V_**2

        vis_ = np.where(Z <= 1.)

        return vis_
