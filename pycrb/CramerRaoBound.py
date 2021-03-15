import numpy as np
import scipy.stats

from pycrb import *


class CramerRaoBound(object):
    """Cramér-Rao bound (CRB) class.

    Attributes:
        K: Snapshots.
        D: Number of superimposed signals.
        arrayManifold: ArrayManifold object.
        uv: UV object.
    """

    def __init__(self, K_, D_, array: ArrayManifold, uv_: UV):

        self.K = K_
        self.D = D_
        self.arrayManifold = array
        self.uv = uv_

    def generate_correlation_matrix(self, Sigma_, df_):
        """Generates a signal spectral matrix.

        Loads diagonal elements of matrix with chi-square variates while a
        complex normal variate is also used. The resulting matrix is a sample
        complex Wishart-distributed matrix.

        Args:
          Sigma_: True covariance matrix.
          df_: Degrees of freedom.

        Returns:
          A sample signal spectral matrix.
        """

        # True covariance matrix
        Sigma = Sigma_

        # Degrees of freedom
        df = df_

        # Obtain dimension of covariance matrix
        dim = Sigma.shape[0]

        # Pre-allocate
        S = np.zeros((dim, dim), dtype=np.complex128)

        # Loading of diagonal elements with square root of chi-square variates
        for n in range(dim):

            df_chi = df - n + 1
            S[n, n] = scipy.stats.chi.rvs(df_chi)

            for j in range(n + 1, dim):

                S[n, j] = scipy.stats.norm.rvs(
                    size=1) + 1j * scipy.stats.norm.rvs(size=1)

        W = np.matmul(S.conj().T, S)

        # Get UT Cholesky factor of Sigma
        R = np.linalg.cholesky(Sigma)

        C = np.matmul(R.conj().T, W).dot(R)

        return C

    def crb(self, sigma_w, wavelength, *args):
        """Performs the computation of the Cramér-Rao bound.

        Args:
          sigma_w: Noise power spectral density.
          wavelength: Wavelength.
          *args: Variable arguments including true covariance matrix and
          degrees of freedom.

        Returns:
          The resulting inverse of the Cramér-Rao bound.
        """

        S_f = self.generate_correlation_matrix(args[0], args[1])

        # Array manifold matrix V
        V_ = np.empty((self.arrayManifold.N, self.D), dtype=np.complex128)

        # Generate D arbitrary plane-wave signals.
        DOAs = args[2]

        for d_ in range(self.D):

            ind_u = np.unravel_index(
                np.argmin(np.abs(DOAs[d_, 0] - self.uv.u)), self.uv.u.shape)

            ind_v = np.unravel_index(
                np.argmin(np.abs(DOAs[d_, 0] - self.uv.v)), self.uv.u.shape)

            V_[:, d_] = np.squeeze(self.arrayManifold.array_manifold_vector(
                wavelength, self.uv)[:, ind_u, ind_v])

        # Derivative matrix D_ (N x 2D)
        D_ = np.empty((self.arrayManifold.N, 2 * self.D), dtype=np.complex128)

        for d_ in range(self.D):

            ind_u = np.unravel_index(
                np.argmin(np.abs(DOAs[d_, 0] - self.uv.u)), self.uv.u.shape)

            ind_v = np.unravel_index(
                np.argmin(np.abs(DOAs[d_, 0] - self.uv.v)), self.uv.u.shape)

            v_u = self.arrayManifold.array_manifold_vector(
                wavelength, self.uv)[:, ind_v, (ind_u[0] - 1):(ind_u[0] + 2)]

            v_v = self.arrayManifold.array_manifold_vector(
                wavelength, self.uv)[:, (ind_v[0] - 1):(ind_v[0] + 2), ind_u]

            deltau = np.abs(self.uv.u[1] - self.uv.u[0])

            deltav = np.abs(self.uv.v[1] - self.uv.v[0])

            col_i = 2 * d_
            col_j = 2 * d_ + 2
            D_[:, col_i:col_j] = np.squeeze(np.array(
                [(-.5 * v_u[:, :, 0] + .5**v_u[:, :, -1]) / deltau, (-.5 * v_v[:, 0, :] + .5**v_v[:, -1, :]) / deltav])).T

        # Yau and Bresler 1992
        C = np.matmul(
            D_.conj().T,
            np.eye(
                self.arrayManifold.N) -
            np.matmul(
                V_,
                np.linalg.inv(
                    np.matmul(
                        V_.conj().T,
                        V_))).dot(
                V_.conj().T)).dot(D_)

        CRBInv = 2 * self.K / sigma_w * \
            np.real(np.multiply(C, np.kron(S_f.T, np.ones((2, 2)))))

        return CRBInv, S_f

    def _matrix_inversion_lemma(self, A_, B_, C_, D_):
        """Performs the matrix inverse of (A+BCD) using the matrix inversion
        lemma.

        Args:
          A, B, C, D: Square matrices (np.array())

        Returns:
          The inverse of (A+BCD), i.e., inv(A+BCD).
        """

        return np.linalg.inv(A_) - np.matmul(
            np.matmul(
                np.linalg.inv(A_),
                B_).dot(
                np.linalg.inv(
                    np.matmul(
                        D_,
                        np.linalg.inv(A_)).dot(B_) + np.linalg.inv(C_))),
            D_).dot(
            np.linalg.inv(A_))
