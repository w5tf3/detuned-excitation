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

# _, s, t, _, _ = am_twocolor_fortran(tau1=2400, tau2=3040, area1=22.65*np.pi, area2=19.29*np.pi, t02=-730, detuning=-8.0000, detuning2=-19.1630)
# _, s2, t2, _, _ = am_twocolor_fortran(phase=np.pi/2,tau1=2400, tau2=3040, area1=22.65*np.pi, area2=19.29*np.pi, t02=-730, detuning=-8.0000, detuning2=-19.1630)

# _, s, t2, _, _ = am_twocolor_fortran(dt=1, tau1=2405, tau2=3035, area1=3*np.pi, area2=0*np.pi, t02=0, detuning=0.0000, detuning2=0.0)
# _, s, t2, _, _ = am_twocolor_fortran(dt=1, tau1=2405, tau2=3035, area1=1*np.pi, area2=4*np.pi, t02=0, detuning=0.2000, detuning2=-7)
# _, s, t2, _, _ = am_twocolor_fortran(dt=1, tau1=4000, tau2=4000, area1=0.91*50*np.pi, area2=2*np.pi, t02=0, detuning=-6, detuning2=-17.385)
# _, s, t2, _, _ = am_twocolor_fortran(dt=1, tau1=4000, tau2=4000, area1=5*np.pi, area2=0*np.pi, t02=0, detuning=-6, detuning2=-17.385)
#s = x2[:,0].real
#p = x2[:,1]
# t=t2
# pulse1 =  p_.pulse1.get_envelope(t)
# pulse2 =  p_.pulse2.get_envelope(t)
# norm = np.max(np.concatenate((pulse1,pulse2)))
# pulse1 = pulse1/norm
# pulse2 = pulse2/norm

# helper.export_csv("data_twocolor2.csv", t, s,pulse1, pulse2, pulse1+pulse2)
# plt.plot(t,s)
# plt.plot(t,s2, 'r-')
# # plt.plot(t3, x3[:,0].real)
# # plt.plot(t,np.real(p), 'r-')
# plt.ylim([-0.1,1.1])
# plt.show()
# det1 = -8
# det2 = -19.163
t,x,p = am_twocolor(dt=1,tau1=2400, tau2=3040, area1=22.65*np.pi, area2=19.29*np.pi, t02=0, detuning=-8.0000, detuning2=-19.1630)
plt.plot(t,x[:,0].real)
plt.show()
# plt.plot(t,x[:,1].real, 'r-')
# plt.plot(t,x[:,1].imag, 'b-')
# plt.show()

# ft = np.fft.fft(x[:,0].real)
# ft2 = np.fft.fft(x[:,1].real)
# ft3 = np.fft.fft(x[:,1].imag)
# dt = np.abs(t[0]-t[1])
# fft_freqs = 2*np.pi*HBAR*np.fft.fftfreq(len(t),d=dt)
# plt.plot(np.fft.fftshift(fft_freqs),np.abs(np.fft.fftshift(ft)), label="ft(f)")
# plt.plot(np.fft.fftshift(fft_freqs),np.abs(np.fft.fftshift(ft2)),label="ft(pr)")
# plt.plot(np.fft.fftshift(fft_freqs),np.abs(np.fft.fftshift(ft3)),label="ft(pi)")
# plt.legend()
# plt.show()
# print(t.shape)

# _, s, t, _,_ = am_twocolor_biexciton(tau1=2400, tau2=3040, area1=22.65*np.pi, area2=19.29*np.pi, t02=0, detuning=-8.0000, detuning2=-19.1630, delta_b=-8)
# _, s, t, _,_ = am_twocolor_biexciton(tau1=2400, tau2=3040, area1=22.65*np.pi, area2=35*np.pi, t02=0, detuning=-8.0000, detuning2=-17.9592, delta_b=3)
# plt.plot(t, s[:,0],label="g")
# plt.plot(t, s[:,1],label="f")
# plt.plot(t, s[:,2],label="b")
# plt.legend()
# plt.show()

# det2s = np.linspace(-6,-16, 100)
# area1 = 25*np.pi
# areas2 = np.linspace(0,2,200)*area1
# p1 = pulse.Pulse(tau=3000, e_start=0, w_gain=0, e0=area1, t0=0)
# rf = lambda t: np.sqrt((p1.get_envelope_f()(t))**2)
# rf_max = rf(t=0)
# print(HBAR*rf_max)
# det1=-5
# print(get_detuning2(tau1=3000,area1=area1,detuning=det1))
# print(get_detuning2(tau1=3000,area1=area1,detuning=det1)-det1)
# detuning_area(det2s, areas2, det1, tau1=3000, tau2=3000, area1=area1, dt=4,t02=0)


# t,x,p = am_twocolor(dt=4,tau1=3000, tau2=2500, area1=50*np.pi, area2=33.84*np.pi, t02=0, detuning=-10.0000, detuning2=-19.101)
# t,x,p = am_twocolor(dt=4,tau1=3000, tau2=2500, area1=0*np.pi, area2=2*np.pi, t02=0, detuning=-5, detuning2=-0)
# t,x,p = am_twocolor(dt=4,tau1=3000, tau2=3000, area1=25*np.pi, area2=32*np.pi, t02=0, detuning=-5, detuning2=-10)
# plt.plot(t,x[:,0].real)
# plt.show()
# print(get_detuning2(3000,19.99,-9)+9)

# t,x,p = am_twocolor(dt=4,tau1=3000, tau2=2500, area1=25*np.pi, area2=3.03*np.pi, t02=0, detuning=-5.0000, detuning2=2.11)
# plt.plot(t,x[:,0].real)
# plt.show()

# detuning1=-10
# delta_b=4
# area1=30*np.pi
# biexciton_stability_pulse2(tau1=2400,tau2=3040,area1=area1,area2s=np.linspace(0.5*area1,2*area1,50),detuning=detuning1,detuning2s=np.linspace(2.5*detuning1,1.8*detuning1,50),delta_b=delta_b)

# _, s, t, _,_ = am_twocolor_sixls(tau1=2400, tau2=3040, area1=22.65*np.pi, area2=19.29*np.pi, t02=0, detuning=-8.0000, detuning2=-19.1630, bz=-4)
# plt.plot(t, s[:,0],label="g")
# plt.plot(t, s[:,1],label="dp")
# plt.plot(t, s[:,2],label="bp")
# plt.plot(t, s[:,3],label="bm")
# plt.plot(t, s[:,4],label="dm")
# plt.plot(t, s[:,5],label="b")
# plt.legend()
# plt.show()

# t2,x2,p = am_twocolor(x0=x[-1], tau1=2400, tau2=3040, area1=22.65*np.pi, area2=19.29*np.pi, t02=0, detuning=-8.0000, detuning2=-19.1630)
# plt.plot(t,x[:,0].real)
# plt.show()
# plt.plot(t,x[:,1].real, 'r-')
# plt.plot(t,x[:,1].imag, 'b-')
# plt.show()
# t,x,_ = am_twocolor(tau1=3040, tau2=3040, area1=1*np.pi, area2=0, t02=0, detuning=0, detuning2=0)
# plt.plot(t,x[:,0].real)
# plt.show()
# plt.plot(t,x[:,1].real, 'r-')
# plt.plot(t,x[:,1].imag, 'b-')
# plt.show()

# phases = np.linspace(0,2*np.pi,50)
# endvals = np.empty_like(phases)
# for i in tqdm.trange(len(phases)):
#     # _, s, t2, _, _ = am_twocolor_fortran(phase=phases[i],tau1=2400, tau2=3040, area1=22.65*np.pi, area2=19.29*np.pi, t02=-730, detuning=-8.0000, detuning2=-19.1630)
#     t,x,p = am_twocolor(phase=phases[i],tau1=2400, tau2=3040, area1=22.65*np.pi, area2=19.29*np.pi, t02=-730, detuning=-8.0000, detuning2=-19.1630)
#     endvals[i] = s[-1]

# plt.plot(phases/np.pi,endvals)
# plt.ylim(-0.1,1.1)
# plt.show()

# areas = np.linspace(22.65*0.9,22.65*1.1,50)*np.pi
# endvals = np.empty_like(areas)
# for i in range(len(areas)):
    # _,s,_,_,_ = am_twocolor_fortran(tau1=2400, tau2=3040, area1=areas[i], area2=19.29*np.pi, t02=-730, detuning=-8.0000, detuning2=-19.1630)
    # endvals[i] = s[-1]
# plt.plot(areas/np.pi,endvals)
# plt.xlabel("area/pi")
# plt.ylabel("occupation")
# plt.show()

# area2 = 22*np.pi
# tau2 = 3035
# tau1 = 2400
# old_max = area2/np.sqrt(2*np.pi*tau2**2)
# new_area = old_max * 8 * tau1
# print(new_area/np.pi)

# t = np.linspace(-4000,4000,300)
# y = pulse.SmoothRectangle(4000,0,e0=2)
# plt.plot(t,y.get_envelope(t))
# plt.show()

# t, x, p = am_first_cw(dt=4, tau1=15000, tau2=3000, e01=3, area2=25*np.pi, detuning=-4.0,onoff=500)
# plt.plot(t, x[:,0].real)
# plt.ylim((0,1))
# plt.show()

# t, x, p = am_second_pulse_cw(n_tau=10,dt=4,tau1=2400, area1=0*np.pi, area2=10/8*33*np.pi, detuning=-5.0, detuning2=-5)
# plt.plot(t, x[:,0].real)
# plt.ylim((0,1))
# plt.show()

# tau1=2400
# area2 = 33*np.pi/(8*tau1)
# det = -7.5327
# area2 = 42.211*np.pi/(8*tau1)
# print(HBAR*area2)
# area2 = 4.55/HBAR 
# f, states, t, polars, energy_pulse2 = am_cw_fortran(dt=1,tau1=tau1, area1=20*np.pi, area2=area2, detuning1=det, t02=-8e3, slope=0.5e3, n_tau=5)
# print(energy_pulse2)
# plt.plot(t,states)
# plt.show()

f, states, t, polars, energy_pulse2 = am_cw_fortran(dt=1,tau1=2400, area1=20*np.pi, area2=3.029/HBAR, detuning1=-5, detuning2=-13.188, t02=-8e3, slope=2e-3, n_tau=5)
print(energy_pulse2)
plt.plot(t, 0.8*np.exp(-0.5*t**2/(2.4e6)),label="gauss (arb.)")
plt.plot(t, 0.5/(1+np.exp(-2e-3*(t+8e3))),label="cw (arb.)")
plt.plot(t,states)
plt.xlabel("t (fs)")
plt.ylabel("X occupation")
plt.legend()
plt.show()

# det2s = np.linspace(-10,-15,70)
# areas_cw = np.linspace(1,6,70)/HBAR
# x,y,z = cw_detuning2_area2(det2s, areas_cw, detuning1=-5, tau1=2400, area1=20*np.pi, t02=-8e3, slope=2e-3, n_tau=5)

# det1s = np.linspace(-4,-7,70)
# area1s = np.linspace(0,25,70)*np.pi
# x,y,z = cw_detuning1_area1(det1s, area1s, area_cw=3/HBAR, detuning2=-13.2, tau1=2400, t02=-8e3, slope=2e-3, n_tau=5)

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
