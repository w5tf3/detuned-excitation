from detuned_excitation.two_level_system import tls_commons, pulse
import numpy as np
import matplotlib.pyplot as plt

HBAR = 6.582119514e2  # meV fs

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

# test_python()
# test_chirp_a()
# test_chirp()