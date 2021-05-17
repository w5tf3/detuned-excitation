from detuned_excitation.amplitude_modulation.am import HBAR
from detuned_excitation.frequency_modulation.fm import fm_pulsed_excitation, fm_rect_pulse
from detuned_excitation.two_level_system.helper import export_csv
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import os
import tqdm

# t, x1, p1 = fm_pulsed_excitation(tau=9000, area=4*np.pi, detuning=-3, small_detuning=1.5)
# t, x1, p1 = fm_rect_pulse(3000, dt=0.5, area=5*np.pi, detuning=-12, small_detuning=4)

def blochsphere(tau, dt, area, detuning, small_detuning, low_z=0.97, mesh=False, rect=False, video=False, export_data=False, rect_modul=False):
    t, x1, p1 = 0,0,0
    if not rect:
        t, x1, p1 = fm_pulsed_excitation(tau=tau, dt=dt, area=area, detuning=detuning, small_detuning=small_detuning)
    if rect:
        t, x1, p1 = fm_rect_pulse(t_0=tau/2, tau=tau, dt=dt, area=area, detuning=detuning, small_detuning=small_detuning, rect_modul=rect_modul)
    

    plt.plot(t,x1[:,0].real)
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
    bloch_x = 2*x1[:,1].real
    bloch_y = -2*x1[:,1].imag
    bloch_z = 2*x1[:,0].real - 1

    # rotation vector
    # remember this one is for a rotating frame with laser frequency.
    rot_x = np.array([-p1.get_envelope(t_) for t_ in t])
    rot_y = 0
    rot_z = np.array([-p1.get_frequency(t_) for t_ in t])

    norm = np.sqrt(rot_x**2 + rot_y**2 + rot_z**2)
    rot_x /= norm
    rot_y /= norm
    rot_z /= norm 

    ax.plot(bloch_x, bloch_y, bloch_z)

    print(plt.cm.jet(x1[-1,0].real))
    print(x1[-1,0].real)

    if export_data:
        pulse1 =  [p1.get_envelope(t_) for t_ in t]
        norm = np.max(pulse1)
        pulse1 = np.array(pulse1)/norm
        detunings = np.array([p1.get_frequency(t_) for t_ in t])*HBAR
        # color data
        colors = []
        for v in x1[:,0].real:
            # this is a 4-tuple with r,g,b,a components
            c = plt.cm.rainbow(v)
            # gnuplots alpha channel is inversed
            color = 65536*int(c[0]*255) + 256*int(c[1]*255) + int(c[2]*255) #+ (c[3])*16777216
            # color = 65536*0 + 256*127 + 255*255
            colors.append(color)
        colors = np.array(colors)
        colors2 = []
        # color_low = int(65536*227 + 256*38 + 54)
        color_low = int(65536*255 + 256*128 + 0)
        color_high = int(65536*255*0 + 256*255*0.498 + 255*1)
        for v in rot_z:
            if v < low_z:
                colors2.append(color_low)
            else:
                colors2.append(color_high)
        colors2 = np.array(colors2, dtype=np.int32)
        # write data to file
        export_csv("data.csv", t, x1[:,0].real, bloch_x, bloch_y, bloch_z, rot_x, rot_y, rot_z, colors, colors2, pulse1, detunings)
    
# rotation axis vector
# ax.quiver(0,0,0,rot_x[0],rot_y[0],rot_z[0])

# for colored version.  this takes a long time 
# for i in range(len(bloch_z)-1):
#     ax.plot(bloch_x[i:i+2], bloch_y[i:i+2], bloch_z[i:i+2],color=plt.cm.coolwarm(0.5*bloch_z[i]+0.5))

    plt.show()

    if video: 
        dir = os.path.dirname(__file__)
        print(dir)
        j = 0
        fig = plt.figure()
        for i in tqdm.trange(1,len(t)-100,100):
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
            ax.quiver(0 ,0 ,0, rot_x[i+50], rot_y[i+50], rot_z[i+50], color='tab:orange')
            plt.savefig(dir+"/pics/image_{}.png".format(j))

#blochsphere(3000, dt=0.5, area=5*np.pi, detuning=-12, small_detuning=4, rect=True, export_data=True)
blochsphere(3500, dt=0.5, area=4.95*np.pi, detuning=-12, small_detuning=4, low_z=0.97, rect=True, export_data=True, rect_modul=True)
#blochsphere(3000, dt=4, area=np.pi, detuning=0.3, small_detuning=0, mesh=True)