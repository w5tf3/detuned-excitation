from detuned_excitation.two_level_system import tls_commons, pulse
import numpy as np
import matplotlib.pyplot as plt

HBAR = 6.582119514e2  # meV fs
HNOBAR = 2*np.pi*HBAR

"""
this file contains some general functions for
chirped excitation of a two level system (adiabatic rapid passage)
the equations are solved using rk4 in python or fortran (a lot faster)
"""

def test_chirp():
    tau = 400
    area = 8*np.pi
    dt = 20
    dt_small = 0.1
    t = np.arange(-4*tau, 4*tau, dt)
    f = np.zeros([len(t)])
    f_ = 0
    p_ = 0+0j
    chirp = 60/(1000**2)*HBAR
    # careful: if chirp*t_ is given as frequency, this has to be multiplied by a factor of 0.5
    # because the resulting frequency for some time t_ has to read w_0 + 0.5 * a * t
    for i in range(len(t)-1):
        t_ = t[i]
        f_, p_ = tls_commons.two_level(t_,dt_small,t_+dt,f_,p_,0.5*chirp*t_,area,tau)
        f[i+1] = f_
    plt.plot(t,f)
    plt.show()

def test_chirp_a():
    """
    directly specify a chirp parameter a in 1/ps^2 
    """
    t0 = 4*400
    dt = 20
    t = np.arange(-t0,t0,dt)
    _,_,states = tls_commons.two_level_chirp_a(t0=-4*400, dt=dt, t_end=4*400-dt, f_start=0., p_start=0.+0.j, energy=0, a_chirp=60/(1000**2), area=8*np.pi, pulse_tau=400)
    plt.plot(t,states)
    plt.show()

test_chirp_a()

def test_twopulse():
    """
    directly specify a chirp parameter a in 1/ps^2 
    """
    tau = 5000
    t0 = 4*tau
    dt = 10
    area1 = 2*np.pi
    t = np.arange(-t0,t0,dt)
    
    _,_,states = tls_commons.twopulse(t0=-t0, dt=dt, t_end=t0-dt,area1=area1, tau1=tau, chirp1=0, energy1=-5)
    plt.plot(t,states)
    plt.title("5ps, 2pi, -5meV")
    plt.xlabel("t in fs")
    plt.ylabel("Besetzung")
    plt.tight_layout()
    plt.show()

# test_twopulse()

def test_python(tau=400, e_start=0, w_gain=60/(1000**2), e0=7*np.pi):
    """
    chirped pulse with python implementation
    """
    p = pulse.Pulse(tau=tau, e_start=e_start, w_gain=w_gain, e0=e0)
    print(p.get_energies())
    x0 = np.array([0,0],dtype=complex)
    t, x2 = tls_commons.runge_kutta(-4*tau, x0, 4*tau, 1, tls_commons.bloch_eq, p, 0)
    # plt.plot(t,x2[:,0].real)
    # plt.show()
    return t, x2

def test_arbitrary(tau=3000, dt=5, area=1*np.pi, energy=0):
    t0 = 5*tau
    t = np.arange(-t0,t0+dt,dt)
    n_steps = len(t)
    e_length = 2*n_steps - 1
    t_new = np.linspace(-t0,t0,e_length)
    _p = pulse.Pulse(tau=tau, e_start=energy, w_gain=0, e0=area)
    e0 = _p.get_total(t_new)
    f,p,states,polars = tls_commons.tls_arbitrary_pulse(-t0, e0, n_steps, dt=dt)
    plt.plot(t, states)
    plt.show()

def elec_field(t, amplitude, tau, t02, frequency):
    """
    frequency as angular frequency omega
    """
    return amplitude / (np.sqrt(2*np.pi)*tau) * np.exp(-(t-t02)**2/(2*tau**2)) * np.exp(-1j*frequency*t)

def test_arbitrary_super(dt=5, tau1=2400, tau2=3040, area1=22.65*np.pi, area2=19.29*np.pi, t02=0, detuning1=-8.0000, detuning2=-19.1630):
    t0 = 8*tau2
    t = np.arange(-t0,t0+dt,dt)
    n_steps = len(t)
    e_length = 2*n_steps - 1
    t_new = np.linspace(-t0,t0,e_length)
    dt_new = np.abs(t_new[0]-t_new[1])
    p1 = pulse.Pulse(tau=tau1, e_start=detuning1, w_gain=0, e0=area1)
    p2 = pulse.Pulse(tau=tau2, e_start=detuning2, w_gain=0, e0=area2,t0=t02)
    #e01 = elec_field(t_new, area1, tau1, 0, detuning1/HBAR)  #p1.get_total(t_new)
    #e02 = elec_field(t_new, area2, tau2, 0, detuning2/HBAR)  #p2.get_total(t_new)
    e01 = p1.get_total(t_new)
    print("max rabi:{:.5f}, max energy:{:.5f}".format(np.abs(np.max(e01)), HBAR*np.abs(np.max(e01))))
    e02 = p2.get_total(t_new)
    pulse_total = e01+e02
    plt.plot(t_new,np.abs(pulse_total))
    plt.show()
    f = np.fft.fft(pulse_total)
    f = np.fft.fftshift(f)
    fft_freqs = 2*np.pi*HBAR*np.fft.fftfreq(len(pulse_total),d=dt_new)
    fft_freqs = np.fft.fftshift(fft_freqs)
    #print(2*np.pi*HBAR/tau1)
    #tau1_spectral = 2*np.pi*HBAR/tau1
    plt.plot(fft_freqs,np.abs(f))
    plt.plot(fft_freqs,(1/dt_new)*area1*np.exp(-0.5*((fft_freqs+detuning1)*(tau1/HBAR))**2))
    print("spectral sigma1:{:.5f}meV".format(HBAR/tau1))
    print("spectral sigma2:{:.5f}meV".format(HBAR/tau2))
    plt.xlim(0,30)
    plt.show()
    f,p,states,polars = tls_commons.tls_arbitrary_pulse(-t0, e01+e02, n_steps, dt=dt)
    plt.plot(t, states)
    plt.show()

# test_arbitrary()
test_arbitrary_super(dt=2)
test_arbitrary_super(dt=2, tau1=1550, tau2=1550, area1=13.06*np.pi, area2=10.45*np.pi, t02=0, detuning1=-5.0,detuning2=-13.5624)
# test_python()
# test_chirp_a()
# test_chirp()

def test_arbitrary_biex(dt=5, tau1=2400, tau2=3040, area1=1*np.pi, area2=0, t02=0, detuning1=0, detuning2=0, delta_b=4):
    t0 = 8*tau2
    t = np.arange(-t0,t0+dt,dt)
    n_steps = len(t)
    e_length = 2*n_steps - 1
    t_new = np.linspace(-t0,t0,e_length)
    dt_new = np.abs(t_new[0]-t_new[1])
    p1 = pulse.Pulse(tau=tau1, e_start=detuning1, w_gain=0, e0=area1)
    p2 = pulse.Pulse(tau=tau2, e_start=detuning2, w_gain=0, e0=area2,t0=t02)
    #e01 = elec_field(t_new, area1, tau1, 0, detuning1/HBAR)  #p1.get_total(t_new)
    #e02 = elec_field(t_new, area2, tau2, 0, detuning2/HBAR)  #p2.get_total(t_new)
    e01 = p1.get_total(t_new)
    print("max rabi:{:.5f}, max energy:{:.5f}".format(np.abs(np.max(e01)), HBAR*np.abs(np.max(e01))))
    e02 = p2.get_total(t_new)
    pulse_total = e01+e02
    plt.plot(t_new,np.abs(pulse_total))
    plt.show()
    f = np.fft.fft(pulse_total)
    f = np.fft.fftshift(f)
    fft_freqs = 2*np.pi*HBAR*np.fft.fftfreq(len(pulse_total),d=dt_new)
    fft_freqs = np.fft.fftshift(fft_freqs)
    #print(2*np.pi*HBAR/tau1)
    #tau1_spectral = 2*np.pi*HBAR/tau1
    plt.plot(fft_freqs,np.abs(f))
    plt.plot(fft_freqs,(1/dt_new)*area1*np.exp(-0.5*((fft_freqs+detuning1)*(tau1/HBAR))**2))
    print("spectral sigma1:{:.5f}meV".format(HBAR/tau1))
    print("spectral sigma2:{:.5f}meV".format(HBAR/tau2))
    plt.xlim(0,30)
    plt.show()
    f,p,states,polars = tls_commons.biex_arbitrary_pulse(e01+e02, n_steps, dt=dt,delta_b=delta_b)
    plt.plot(t, states[:,1],label="X")
    plt.plot(t, states[:,2],label="XX")
    plt.legend()
    plt.show()

test_arbitrary_biex()
test_arbitrary_biex(detuning1=-2,area1=4.6154*np.pi)
