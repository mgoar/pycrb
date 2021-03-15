import unittest
import numpy as np
import matplotlib.pyplot as plt

from pycrb import ArrayManifold, CramerRaoBound, UV

# array manifold parameters
N = 9    # array elements
c = 299792458     # [m/s]
FREQ = 40e9       # [Hz]
wl = c / FREQ

d_ = 0.5 * wl


class TestCramerRaoBound(unittest.TestCase):

    def test_correlation_matrix(self):

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

        testCramerRao = CramerRaoBound(
            100, 1, testArrayManifold, testUv)

        # Wishart method
        R = np.array([[1., 0.1], [0.1, 0.71]])

        testS_f = testCramerRao.generate_correlation_matrix(R, 3)

        self.assertEqual(testS_f.shape[0], 2)

        self.assertEqual(testS_f.shape[1], 2)

        # check for positive-definiteness
        self.assertTrue(np.linalg.det(testS_f) != 0)

    def test_crb(self):

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

        testCramerRao = CramerRaoBound(
            100, 2, testArrayManifold, testUv)

        testCrb, testS_f = testCramerRao.crb(1, wl, np.array([[1., 0.1], [0.1, 0.35]]), 5, np.array([[0.25, 0.1], [-0.25, 0.1]]))

        self.assertEqual(testCrb.shape[0], 4)

        self.assertEqual(testCrb.shape[1], 4)
        
        # n=1, N=1 and ULA (Stoica abd Nehorai, 1989)
        N_ULA = 3
        testArrayManifold = ArrayManifold(N_ULA, np.ones(
            N_ULA), X.flatten()[0:N_ULA], Y.flatten()[0:N_ULA], 101, 101)
            
        testUv = UV(128, 128)
    
        testCramerRao = CramerRaoBound(
            100, 1, testArrayManifold, testUv)
            
        L = 1000
        SNR = []
        ULACRB = []
        for l in range(L):
            
            testCrb, S_f = testCramerRao.crb(0.1, wl, np.array([[1.]]), 5, np.array([[.25]]))
            
            ULACRB.append(testCrb[0][0])
            SNR.append(np.real(S_f[0][0])/np.sqrt(0.1))
            
        histULACRBdB = np.histogram(10*np.log10(np.power(ULACRB,-1)))
        histSNRdB = np.histogram(10*np.log10(SNR))
        
        ulaCRBdB = histULACRBdB[1][np.argmax(histULACRBdB[0])]
        snrdB = histSNRdB[1][np.argmax(histSNRdB[0])]
        
        self.assertTrue(np.isclose(ulaCRBdB, 10*np.log10(6/(100**3*10**(snrdB/10))/100), atol=3))


    def test_matrix_inversion_lemma(self):

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

        testCramerRao = CramerRaoBound(
            100, 2, testArrayManifold, testUv)

        invTest = testCramerRao._matrix_inversion_lemma(np.array([[1.5, 0.125], [-0.5, 0.125]]),
                                                        np.array([[-0.4, 0.1], [0.275, 0.1125]]),
                                                        np.array([[0.0, 1.0], [0.15, 0.0]]),
                                                        np.array([[1.125, 0.5], [-0.1, -0.1]]))

        self.assertAlmostEqual(invTest[0][0], 0.4193, places=5)
        self.assertAlmostEqual(invTest[0][1], -0.68276, places=5)
        self.assertAlmostEqual(invTest[1][0], 2.01272, places=5)
        self.assertAlmostEqual(invTest[1][1], 6.16217, places=5)


if __name__ == '__main__':
    unittest.main()
