from detuned_excitation.amplitude_modulation.am import * # am_twopulse_excitation, test_beat, test_beat_special_frame1, test_beat_special_frame2
from detuned_excitation.two_level_system import pulse, helper
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import os
import tqdm

HBAR = 6.582119514e2  # meV fs

# t2, x2, p_ = test_beat(dt=1, tau1=6200, tau2=9600, area1=29.0*np.pi, area2=29.0*np.pi, detuning=-5.0, t02=0) # t02=-1800)
# t2, x2, p_ = test_beat(tau1=2405, tau2=3035, area1=22.4984*np.pi, area2=20.1275*np.pi, t02=-725, detuning=-8.0000)
# t2, x2, p_ = test_beat(dt=1,tau1=2405, tau2=3035, area1=22.31*np.pi, area2=19.90*np.pi, t02=-725, detuning=-8.0000)
# t2, x2, p_ = test_beat(tau1=2400, tau2=3040, area1=22.31*np.pi, area2=19.90*np.pi, t02=-730, detuning=-8.0000)

# t3, x3, p2 = test_beat(tau1=2405, tau2=3035, area1=1*np.pi, area2=0*np.pi, t02=-725, detuning=0.0)
# energy1: -5meV, energy2:-11.3158meV
# _, s, t2, _, _ = am_twopulse_excitation(dt=1, tau1=2405, tau2=3035, area1=22.4984*np.pi, area2=20.1275*np.pi, t02=-6200, detuning=-8.0000, detuning2=0.7)

# t2, x2, p_ = test_beat(dt=1, detuning=-12, tau1=3204.0816, area1=35.1020*np.pi, area2=35.1020*np.pi, tau2=3204.0816, t02=3844.8980)
# t2, x2, p_ = test_beat(dt=1, detuning=-9, tau1=3938.7755, area1=33.4694*np.pi, area2=33.4694*np.pi, tau2=3938.7755, t02=4726.5306)

# weird 3pi case
# _, s, t2, _, _ = am_twocolor_fortran(dt=1, detuning=-5, tau1=2920.7755, area1=35.4694*np.pi, area2=1.5*35.4694*np.pi, tau2=2920.7755, t02=1.5*2920.5306)

_, s, t2, _, _ = am_twocolor_fortran(tau1=2400, tau2=3040, area1=22.65*np.pi, area2=19.29*np.pi, t02=-730, detuning=-8.0000, detuning2=-19.1630)
# _, s, t2, _, _ = am_twocolor_fortran(dt=1, tau1=2405, tau2=3035, area1=3*np.pi, area2=0*np.pi, t02=0, detuning=0.0000, detuning2=0.0)
# _, s, t2, _, _ = am_twocolor_fortran(dt=1, tau1=2405, tau2=3035, area1=1*np.pi, area2=4*np.pi, t02=0, detuning=0.2000, detuning2=-7)
# _, s, t2, _, _ = am_twocolor_fortran(dt=1, tau1=4000, tau2=4000, area1=0.91*50*np.pi, area2=2*np.pi, t02=0, detuning=-6, detuning2=-17.385)
# _, s, t2, _, _ = am_twocolor_fortran(dt=1, tau1=4000, tau2=4000, area1=5*np.pi, area2=0*np.pi, t02=0, detuning=-6, detuning2=-17.385)
#s = x2[:,0].real
#p = x2[:,1]
t=t2
# pulse1 =  p_.pulse1.get_envelope(t)
# pulse2 =  p_.pulse2.get_envelope(t)
# norm = np.max(np.concatenate((pulse1,pulse2)))
# pulse1 = pulse1/norm
# pulse2 = pulse2/norm

# helper.export_csv("data_twocolor2.csv", t, s,pulse1, pulse2, pulse1+pulse2)
plt.plot(t,s)
# plt.plot(t3, x3[:,0].real)
# plt.plot(t,np.real(p), 'r-')
plt.ylim([-0.1,1.1])
plt.show()

# area2 = 22*np.pi
# tau2 = 3035
# tau1 = 2400
# old_max = area2/np.sqrt(2*np.pi*tau2**2)
# new_area = old_max * 8 * tau1
# print(new_area/np.pi)



#t, x, p = am_second_pulse_cw(dt=1,tau1=2400, area1=20*np.pi, area2=33*np.pi, detuning=-5.0)
#plt.plot(t, x[:,0].real)
# plt.ylim((0,1))
#plt.show()

# tau1=2400
# area2 = 33*np.pi/(8*tau1)
# det = -7.5327
# area2 = 42.211*np.pi/(8*tau1)
# f, states, t, polars, energy_pulse2 = am_cw_fortran(dt=1,tau1=tau1, area1=20*np.pi, area2=area2, detuning=det, t02=0)
# plt.plot(t,states)
# plt.show()

# dets = np.linspace(-5,-11,50)
# areas_cw = np.linspace(0,60,50)*np.pi
# x,y,z = cw_detuning_area(dets, areas_cw, tau1=2400, area1=20*np.pi)

# taus = np.linspace(1000,10000,50)
# tau2s = 1.5*taus # + np.linspace(150,4000,50)
# areas = np.linspace(0,40,50)*np.pi
# areas2 = areas
# t02s = 0*taus
# x,y,z = four_parameter_stability(taus, areas, tau2s, areas2, -7.5, t02s)
#helper.save_colormap("data_m11_tau2_times1_5.csv", x/1000,y/np.pi,z)

# stability_factor2()

# rffreq = 0#0.01224
# #plt.plot(t,(x2[:,1]*np.exp(1j*0.0169*t)).imag, 'b-')
# plt.plot(t,(x2[:,1]*np.exp(1j*rffreq*t)).real, 'r-')
# plt.show()



def special_delta1(area1,tau1,area2,tau2,t2,freq1,freq2, n=1000):
    tau = tau1 if tau1 > (tau2+np.abs(t2)) else (tau2+np.abs(t2))
    t0 = -4*tau
    t1 = 4*tau
    t = np.linspace(t0, t1, n)

    omega1 = pulse.Pulse(tau=tau1, e0=area1, e_start=0).get_envelope_f()
    omega2 = pulse.Pulse(tau=tau2, e0=area2, e_start=0, t0=t2).get_envelope_f()

    delta = freq2 - freq1
    delta_1 = (omega2(t2)**2 - omega1(0)**2 + delta**2)/(2*delta)
    delta_1 = np.array([delta_1 for i in t])
    plt.plot(t,omega1(t),label='Omega1')
    plt.plot(t,omega2(t),label='Omega2')
    plt.plot(t,delta_1,label='delta_1')
    plt.legend()
    plt.show()

    plt.plot(t,HBAR*delta_1,label='HBAR*delta_1')
    plt.ylabel("E in meV")
    plt.legend()
    plt.show()

    plt.plot(t,HBAR*(freq1 + delta_1),label='HBAR*omega_rot')
    plt.ylabel("E in meV")
    plt.legend()
    plt.show()

# special_delta1(tau1=6200,tau2=9600,area1=29*np.pi,area2=29*np.pi,freq1=-5/HBAR,freq2=-11.3158/HBAR, t2=-1800)
