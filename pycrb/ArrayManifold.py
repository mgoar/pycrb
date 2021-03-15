import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

from pycrb import UV


class ArrayManifold(object):
    """Array manifold class"""

    def __init__(self, N_, w_, x_, y_, n_theta_, n_phi_):

        self.N = N_
        self.w = w_
        self.x = x_
        self.y = y_
        self.n_theta = n_theta_
        self.n_phi = n_phi_

    def interelement_distance(x_, y_):

        return distance.cdist(x_, y_, 'euclidean')

    def array_manifold_vector(self, wavelength, uv: UV):

        U, V = np.meshgrid(uv.u, uv.v)

        VV = np.asarray([np.exp(1j * 2 * np.pi / wavelength * (xx * U + yy * V))
                         for (xx, yy) in zip(self.x, self.y)])

        return VV

    def conventional_beamformer(self, wavelength, uv: UV, uv_s: UV):

        V = self.array_manifold_vector(wavelength, uv)

        V_s = self.array_manifold_vector(wavelength, uv_s)

        B_temp = np.reshape(
            np.multiply(
                V_s.flatten().conj().T,
                V.flatten()),
            (self.N,
             128,
             128))

        B = np.sum(B_temp, axis=0)

        return B

    def sampling_points(self):

        return self.x[None, :] - \
            self.x[:, None], self.y[None, :] - self.y[:, None]

    def _plot(self, wavelength):

        plt.plot(self.x / wavelength, self.y / wavelength, 'o')
        plt.axis('equal')
        plt.xlabel('x (wavelengths)')
        plt.ylabel('y (wavelengths)')
        plt.grid()
