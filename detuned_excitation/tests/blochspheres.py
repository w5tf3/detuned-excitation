from detuned_excitation.tests.twolevelsystem import test_excitation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

t, x1 = test_excitation(tau=9000, area=4*np.pi, detuning=3, small_detuning=1.5)
plt.plot(t,x1[:,0].real)
plt.show()


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_aspect("auto")

# draw sphere
# u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
# x = np.cos(u)*np.sin(v)
# y = np.sin(u)*np.sin(v)
# z = np.cos(v)
# ax.plot_wireframe(x, y, z, color="gray")
ax.set_xlabel("2Re(p)")
ax.set_ylabel("-2Im(p)")
ax.set_zlabel("2f-1")

bloch_x = 2*x1[:,1].real
bloch_y = -2*x1[:,1].imag
bloch_z = 2*x1[:,0].real - 1

ax.plot(bloch_x, bloch_y, bloch_z)

#for i in range(len(bloch_z)-1):
#    ax.plot(bloch_x[i:i+2], bloch_y[i:i+2], bloch_z[i:i+2],color=plt.cm.coolwarm(0.5*bloch_z[i]+0.5))

plt.show()