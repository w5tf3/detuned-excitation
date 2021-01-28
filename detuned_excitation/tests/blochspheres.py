from detuned_excitation.tests.twolevelsystem import test_excitation
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import os
import tqdm

t, x1, p1 = test_excitation(tau=9000, area=4*np.pi, detuning=-3, small_detuning=1.5)
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
# ax.plot_wireframe(x, y, z, color="lightgray", alpha=0.6)
ax.set_xlabel("2Re(p)")
ax.set_ylabel("-2Im(p)")
ax.set_zlabel("2f-1")

# bloch vector
bloch_x = 2*x1[:,1].real
bloch_y = -2*x1[:,1].imag
bloch_z = 2*x1[:,0].real - 1

# rotation vector
# remember this one is for a rotating frame with laser frequency.
rot_x = -p1.get_envelope(t)
rot_y = 0
rot_z = p1.get_frequency(t)

norm = np.sqrt(rot_x**2 + rot_y**2 + rot_z**2)
rot_x /= norm
rot_y /= norm
rot_z /= norm 

ax.plot(bloch_x, bloch_y, bloch_z)
# rotation axis vector
# ax.quiver(0,0,0,rot_x[0],rot_y[0],rot_z[0])

# for colored version.  this takes a long time 
# for i in range(len(bloch_z)-1):
#     ax.plot(bloch_x[i:i+2], bloch_y[i:i+2], bloch_z[i:i+2],color=plt.cm.coolwarm(0.5*bloch_z[i]+0.5))

plt.show()
dir = os.path.dirname(__file__)
print(dir)
j = 0
# for i in tqdm.trange(1,len(t)-100,100):
#     plt.close(fig)
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.set_aspect("auto")

#     # draw sphere
#     u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
#     x = 0.98*np.cos(u)*np.sin(v)
#     y = 0.98*np.sin(u)*np.sin(v)
#     z = 0.98*np.cos(v)
#     ax.plot_wireframe(x, y, z, color="lightgray", alpha=0.6)

#     ax.set_xlabel("2Re(p)")
#     ax.set_ylabel("-2Im(p)")
#     ax.set_zlabel("2f-1")

#     j += 1
#     ax.plot(bloch_x[:i], bloch_y[:i], bloch_z[:i], color='tab:blue')
#     ax.plot(bloch_x[i:i+50], bloch_y[i:i+50], bloch_z[i:i+50], color='tab:red')
#     ax.quiver(0 ,0 ,0, rot_x[i+50], rot_y[i+50], rot_z[i+50], color='tab:orange')
#     plt.savefig(dir+"/pics/image_{}.png".format(j))
    
