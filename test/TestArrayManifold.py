import unittest
import numpy as np
import matplotlib.pyplot as plt

from pycrb import ArrayManifold, UV

# array manifold parameters
N = 25    # array elements
c = 299792458     # [m/s]
FREQ = 40e9       # [Hz]
wl = c / FREQ

d_ = 0.5 * wl


class TestArrayManifold(unittest.TestCase):

    def test_array_manifold_vector(self):

        # Standard Rectangular Array (SRA) x, y
        x_ = np.linspace(
            start=-
            np.round(
                np.sqrt(N) /
                2) *
            d_,
            stop=np.round(
                np.sqrt(N) /
                2) *
            d_,
            num=int(
                np.sqrt(N)))
        y_ = np.linspace(
            start=-
            np.round(
                np.sqrt(N) /
                2) *
            d_,
            stop=np.round(
                np.sqrt(N) /
                2) *
            d_,
            num=int(
                np.sqrt(N)))

        X, Y = np.meshgrid(x_, y_)

        testArrayManifold = ArrayManifold(N, np.ones(
            N), X.flatten(), Y.flatten(), 101, 101)

        testUv = UV(128, 128)

        V = testArrayManifold.array_manifold_vector(wl, testUv)

        self.assertEqual(V.shape[0], N)
        self.assertEqual(V.shape[1:], (128, 128))

    def test_beam_pattern(self):

        # Standard Rectangular Array (SRA) x, y
        x_ = np.linspace(
            start=-
            np.round(
                np.sqrt(N) /
                2) *
            d_,
            stop=np.round(
                np.sqrt(N) /
                2) *
            d_,
            num=int(
                np.sqrt(N)))
        y_ = np.linspace(
            start=-
            np.round(
                np.sqrt(N) /
                2) *
            d_,
            stop=np.round(
                np.sqrt(N) /
                2) *
            d_,
            num=int(
                np.sqrt(N)))

        X, Y = np.meshgrid(x_, y_)

        testArrayManifold = ArrayManifold(N, np.ones(
            N), X.flatten(), Y.flatten(), 101, 101)

        testUv = UV(128, 128)

        # Conventional delay-and-sum beamformer at boresight
        testUv_s = UV(128, 128)
        testUv_s.steer(.0, .0)
        B = testArrayManifold.conventional_beamformer(wl, testUv, testUv_s)

        self.assertEqual(B.shape, (128, 128))
        self.assertEqual(
            np.unravel_index(
                np.argmax(
                    np.abs(B)), B.shape), (63, 63))

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Plot the surface
        U_, V_ = np.meshgrid(testUv.u, testUv.v)

        ax.plot_surface(U_, V_, 20 * np.log10(np.abs(B)), cmap='viridis',
                        linewidth=0, antialiased=False)

        plt.show()

        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_zlabel('Beam pattern [dB]')

        ax.set_zlim(-30, 20)


if __name__ == '__main__':
    unittest.main()
