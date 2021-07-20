from detuned_excitation.amplitude_modulation.am import *
from detuned_excitation.two_level_system.helper import export_csv
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import os
import tqdm

#f,s,t,p,_ = am_twopulse_excitation(dt=1, tau1=6192, tau2=9583, area1=29.0*np.pi, area2=29.0*np.pi, t02=-1812)
#t2, x2, p_ = test_beat(dt=1, tau1=6200, tau2=9600, area1=29.0*np.pi, area2=29.0*np.pi, t02=-1800)
#t2, x2, p_ = test_beat_special_frame1(dt=1, tau1=6200, tau2=9600, area1=29.0*np.pi, area2=29.0*np.pi, t02=-1800)
# energy1: -5meV, energy2:-11.3158meV
def blochsphere(tau1=2405, tau2=3035, area1=22.4984*np.pi, area2=20.1275*np.pi, t02=-725, detuning=-8.0,detuning2=None, video=False, mesh=False, export_data=None):
    t, x2, p_ = am_twocolor(tau1=tau1, tau2=tau2, area1=area1, area2=area2, t02=t02, detuning=detuning, detuning2=detuning2)

    s = x2[:,0].real
    p = x2[:,1]

    plt.plot(t,s)
    # plt.plot(t,np.real(p), 'r-')
    plt.ylim([-0.1,1.1])
    plt.show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect("auto")

    # draw sphere
    if mesh:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        ax.plot_wireframe(x, y, z, color="lightgray", alpha=0.6)
    ax.set_xlabel("2Re(p)")
    ax.set_ylabel("-2Im(p)")
    ax.set_zlabel("2f-1")

    # bloch vector
    bloch_x = 2*p.real
    bloch_y = -2*p.imag
    bloch_z = 2*s - 1

    ax.plot(bloch_x, bloch_y, bloch_z)
    if export_data is not None:
        export_csv(export_data, t, s, np.real(p), np.imag(p), bloch_x, bloch_y, bloch_z)
        
    # for i in range(len(bloch_z)-1):
    #     ax.plot(bloch_x[i:i+2], bloch_y[i:i+2], bloch_z[i:i+2],color=plt.cm.coolwarm(0.5*bloch_z[i]+0.5))

    plt.show()
    if video:
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
            os.makedirs(os.path.dirname("pics/"), exist_ok=True)
            plt.savefig("pics/image_{}.png".format(j))
