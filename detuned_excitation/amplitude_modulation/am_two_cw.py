import numpy as np
from detuned_excitation.amplitude_modulation.am import am_cw_fortran
from detuned_excitation.two_level_system.pulse import SmoothCW
from detuned_excitation.two_level_system.tls_commons import tls_arbitrary_pulse
from detuned_excitation.two_level_system.helper import export_csv
import matplotlib.pyplot as plt
import tqdm

HBAR = 6.582119514e2  # meV fs



def endvalue_cw2(amp2s, det2s, amp1=1/HBAR, det1=-1, gamma_x=1/100e3, t0=3e3, duration=800e3, dt=0.05e3):
    x_ax = det2s
    y_ax = amp2s
    endvals = np.empty([len(y_ax), len(x_ax)])
    t = np.arange(0,duration,dt)
    n_steps = int((len(t)+1)/2)
    tnew = np.linspace(0,n_steps-1,n_steps)*2*dt
    p1 = SmoothCW(e0=amp1,e_start=det1,t0=t0,alpha_on=0.2e3)
    for i in tqdm.trange(len(y_ax)):
        for j in range(len(x_ax)):
            p2 = SmoothCW(e0=y_ax[i],e_start=x_ax[j],t0=t0,alpha_on=0.2e3)
            field = p1.get_total(t) + p2.get_total(t)
            f,p,states,polars = tls_arbitrary_pulse(t[0], field,n_steps=n_steps,dt=2*dt,strict=False,gamma_x=gamma_x)
            states_mean = states[-int(100e3/(2*dt)):]
            endvals[i,j] = np.sum(states_mean)/len(states_mean)
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    max_x,max_y = x_ax[ind[1]],y_ax[ind[0]]
    print("{}, det2:{:.4f}, t02:{:.4f}, endval:{:.4f}".format(ind,max_x,max_y,endvals[ind[0],ind[1]]))
    plt.xlabel("det2 (meV)")
    plt.ylabel("amplitude (meV)")
    plt.title("mean of 100ps after {}ps, cw1: ampl. {:.1f} meV, det. {}meV, gamma=1/200ps".format(duration*1e-3-100,amp1*HBAR,det1))
    plt.pcolormesh(x_ax, y_ax*HBAR, endvals, shading='auto')
    plt.plot(x_ax[ind[1]],y_ax[ind[0]]*HBAR, 'r.')
    plt.colorbar()
    plt.show()
    return x_ax, y_ax, endvals

amp2s = np.linspace(0,4,50)/HBAR
det2s = np.linspace(-1,-4,50)
# endvalue_cw2(amp2s,det2s, gamma_x=1/200e3,duration=1300e3)

duration = 1300e3
# half the time step of the result
dt = 0.05e3
gamma_x = 0#1/200e3  # 1/fs
t = np.arange(0,duration,dt)

p1 = SmoothCW(e0=1/HBAR,e_start=-1,t0=3e3,alpha_on=0.2e3)
p2 = SmoothCW(e0=2/HBAR,e_start=-2.8,t0=3e3,alpha_on=0.2e3)
# plt.plot(t,p1.get_envelope(t))
# plt.show()

field = p1.get_total(t) + p2.get_total(t)
n_steps = int((len(t)+1)/2)
tnew = np.linspace(0,n_steps-1,n_steps)*2*dt
f,p,states,polars = tls_arbitrary_pulse(t[0], field,n_steps=n_steps,dt=2*dt,strict=False,gamma_x=gamma_x)
states_mean = states[-int(100/(2*dt)):]
print(np.sum(states_mean)/len(states_mean))
# f2, states2, t2, polars2, energy_pulse2 = am_cw_fortran(area1=0,area2=1/HBAR,detuning2=0)
# plt.plot(t2,states2)
plt.plot(tnew*1e-3,states)
plt.xlabel("t in ps")
plt.title("det2=-2.8, amp2=2 mev")
plt.ylabel("x occupation")
plt.ylim(0,1)
# plt.plot(tnew,np.sin(1/(2*HBAR)*tnew)**2,label="sin")
# plt.legend()
plt.show()
