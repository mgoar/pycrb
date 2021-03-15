import numpy as np
import matplotlib.pyplot as plt

from pycrb import *

# array manifold parameters
D = 2    # plane waves
N = 4    # array elements
c = 299792458     # [m/s]
FREQ = 40e9       # [Hz]
l = c / FREQ

d = 0.89 * l
deltau = d / l

yArrayManifold = ArrayManifold(N, np.ones(N), np.array(
    [0.0, -np.sqrt(3) / 2 * d, +np.sqrt(3) / 2 * d, 0.0]), np.array([0.0, d / 2, d / 2, -d]), 101, 101)

u, v = yArrayManifold.sampling_points()
plt.plot(u / l, v / l, '*')
plt.axis('equal')
plt.xlabel('u (wavelengths)')
plt.ylabel('v (wavelengths)')
plt.grid()

# Conventional delay-and-sum beamformer at boresight
uv = UV(128, 128)
uv_s = UV(128, 128)
uv_s.steer(.0, .0)
B = yArrayManifold.conventional_beamformer(l, uv, uv_s)

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface
U_, V_ = np.meshgrid(uv.u, uv.v)

# Visible region
ii = uv._visible_region()

mask = np.zeros(B.shape, dtype=np.complex128)
mask[ii] = 1.

B = np.multiply(B, mask)

ax.plot_surface(U_, V_, 20 * np.log10(np.abs(B)), cmap='viridis',
                linewidth=0, antialiased=False, vmin=-30, vmax=15)

ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('Beam pattern [dB]')

ax.set_zlim(-30, 15)

plt.show()
