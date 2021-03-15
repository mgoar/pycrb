import numpy as np
import collections

from pycrb import *

# array manifold parameters
D = 2    # plane waves
N = 4    # array elements
c = 299792458     # [m/s]
FREQ = 40e9       # [Hz]
l = c / FREQ

d_ = 0.89 * l

# Standard Rectangular Array (SRA) x, y
x_ = np.linspace(
    start=-
    np.round(
        np.sqrt(9) /
        2) *
    d_,
    stop=np.round(
        np.sqrt(9) /
        2) *
    d_,
    num=int(
        np.sqrt(9)))
y_ = np.linspace(
    start=-
    np.round(
        np.sqrt(9) /
        2) *
    d_,
    stop=np.round(
        np.sqrt(9) /
        2) *
    d_,
    num=int(
        np.sqrt(9)))

X, Y = np.meshgrid(x_, y_)

yArrayManifold = ArrayManifold(
    N,
    np.ones(N),
    X.flatten(),
    Y.flatten(),
    101,
    101)

# Conventional delay-and-sum beamformer at boresight
uv = UV(128, 128)

Parameters = collections.namedtuple('Parameters',
                                    ['sigma_w',
                                     'Sigma',
                                     'K',
                                     'D',
                                     'x_',
                                     'y_',
                                     'n_theta',
                                     'n_phi',
                                     'N_',
                                     'w_',
                                     'wavelength'])

myParams = Parameters(0.1, np.array([[1., 0.1], [0.1, 0.35]]), 100, 2, X.flatten(
).tolist(), Y.flatten().tolist(), 101, 101, 9, np.ones(9), l)

arrayOpt = ArrayOptimizer(uv, myParams)

optCoords, nf, rc = arrayOpt.optimize(np.array([[0.5, 0.0], [-0.5, 0.1]]))
