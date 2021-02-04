from beat import test_twopulse
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import os
import tqdm

f,s,t,p = test_twopulse(dt=1, tau1=6192, tau2=9583, area1=29.0*np.pi, area2=29.0*np.pi, t02=-1812)
plt.plot(t,s)
# plt.plot(t,np.real(p), 'r-')
plt.ylim([-0.1,1.1])
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
bloch_x = 2*p.real
bloch_y = -2*p.imag
bloch_z = 2*s - 1

ax.plot(bloch_x, bloch_y, bloch_z)

with open("data.txt", 'w') as f:
    print("writing file")
    for i in range(len(t)):
        f.write("{:.4f} {:.4f} {:.4f} {:.4f}\n".format(t[i],np.real(s[i]), np.real(p[i]), np.imag(p[i])))


# for i in range(len(bloch_z)-1):
#     ax.plot(bloch_x[i:i+2], bloch_y[i:i+2], bloch_z[i:i+2],color=plt.cm.coolwarm(0.5*bloch_z[i]+0.5))

plt.show()
def video_pics():
    dir = os.path.dirname(__file__)
    print(dir)
    j = 0
    for i in tqdm.trange(int(len(t)*0.3),int(len(t)*0.8),int((len(t)*0.5)/200)):
        plt.close(fig)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect("auto")

        # draw sphere
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = 0.98*np.cos(u)*np.sin(v)
        y = 0.98*np.sin(u)*np.sin(v)
        z = 0.98*np.cos(v)
        ax.plot_wireframe(x, y, z, color="lightgray", alpha=0.6)

        ax.set_xlabel("2Re(p)")
        ax.set_ylabel("-2Im(p)")
        ax.set_zlabel("2f-1")

        j += 1
        ax.plot(bloch_x[:i], bloch_y[:i], bloch_z[:i], color='tab:blue')
        ax.plot(bloch_x[i:i+50], bloch_y[i:i+50], bloch_z[i:i+50], color='tab:red')
        plt.savefig(dir+"/pics/image_{}.png".format(j))
