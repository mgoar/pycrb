import numpy as np
import collections
from scipy import optimize

from pycrb import UV, ArrayManifold, CramerRaoBound

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


class ArrayOptimizer(object):
    """Array optimization class"""

    def __init__(self, uv_: UV, params: Parameters):

        self.uv = uv_
        self.N = params.N_
        self.w = params.w_
        self.n_theta = params.n_theta
        self.n_phi = params.n_phi
        self.D = params.D
        self.K = params.K
        self.sigma_w = params.sigma_w
        self.wavelength = params.wavelength
        self.Sigma = params.Sigma

    def objective(self, coords_, DOAs_):
        """Objective function to minimize (variance)"""

        x_ = coords_[0:self.N]

        y_ = coords_[self.N:]

        thisArrayManifold = ArrayManifold(
            self.N, self.w, x_, y_, self.n_theta, self.n_phi)

        thisCramerRaoBound = CramerRaoBound(
            self.K, self.D, thisArrayManifold, self.uv)

        return thisCramerRaoBound.crb(
            self.sigma_w,
            self.wavelength,
            self.Sigma,
            self.K,
            DOAs_)

    def optimize(self, DOAs_):
        """Optimization function"""

        def feval(x0): return np.diag(self.objective(x0, DOAs_))[0]

        d_ = 0.5 * self.wavelength

        x_ = np.linspace(
            start=-
            np.round(
                np.sqrt(self.N) /
                2) *
            d_,
            stop=np.round(
                np.sqrt(self.N) /
                2) *
            d_,
            num=int(
                np.sqrt(self.N)))
        y_ = np.linspace(
            start=-
            np.round(
                np.sqrt(self.N) /
                2) *
            d_,
            stop=np.round(
                np.sqrt(self.N) /
                2) *
            d_,
            num=int(
                np.sqrt(self.N)))

        X, Y = np.meshgrid(x_, y_)

        x_0 = np.hstack((X.flatten(), Y.flatten()))

        bnds = ((-self.wavelength, +self.wavelength),) * self.N * 2

        xopt, nfeval, rc = optimize.fmin_tnc(
            feval, x_0, approx_grad=True, bounds=bnds, xtol=1e-3)

        return xopt, nfeval, rc
