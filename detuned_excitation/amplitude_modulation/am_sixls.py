import numpy as np
import matplotlib.pyplot as plt
from detuned_excitation.two_level_system import tls_commons, pulse, helper
import tqdm

HBAR = 6.582119514e2  # meV fs

def am_twocolor_fortran(tau1=5000, tau2=5000, e0=0, dt=5, area1=10*np.pi, area2=10*np.pi, detuning=-5, t02=0, factor=1.0, factor2=1.0, detuning2=None, polar1=1.0, polar2=1.0, bx=0, bz=0, d0=0.25, delta_b=0.25, x0=np.array([1, 0, 0, 0, 0, 0]), p0=np.zeros([15], dtype=complex),delta_E=0):
    """
    two pulses added together, forming a beat. t02 is the time difference between the two pulses.
    pulse 1 is centered around t=0, pulse 2 around t02.
    the energy of the first pulse is given as a parameter (detuning), the energy of the second pulse
    is calculated according to detuning - factor * max_rabi_freq (of pulse 1) * HBAR
    factor = 1 seems optimal. For some factor, we will have resonant excitation, so watch out if this is not wanted.
    returns endvalue, dynamics, time, polarization dynamics, energy of second pulse

    """
    # take a time window which fits both pulses, even if one is centered around t != 0
    tau = tau1 if tau1 > (tau2+np.abs(t02)) else (tau2+np.abs(t02))
    t0 = 4*tau
    # t = np.arange(-t0,t0,dt)

    # here we calculate the laser frequency for the second pulse
    p1 = pulse.Pulse(tau=tau1, e_start=detuning, w_gain=0, e0=area1, t0=0)
    rf = lambda t: np.sqrt(factor2*(p1.get_envelope_f()(t))**2 + (detuning/HBAR)**2)
    rf_max = rf(t=0)  # max of rabifreq
    # the energy of the second laser should be the same as the first one, but reduced by rabifrequency*HBAR
    # factor = 1 seems to be optimal
    energy_pulse2 = detuning - factor * HBAR*rf_max
    if detuning2 is not None:
        energy_pulse2 = detuning2
    # E_dp, E_bp, E_bm, E_dm, E_b = helper.energies(bz=bz,delta_B=delta_b,delta_E=0.0,d0=d0)
    e1 = e0 + detuning
    e2 = e0 + energy_pulse2
    
    # print("energy1: {:.4f}meV, energy2: {:.4f}meV".format(detuning, energy_pulse2))

    t, s, ends, endpol, pol = tls_commons.six_levels_two_color(-t0, t0, dt=dt, tau1=tau1, tau2=tau2, energy_1=e1, energy_2=e2, e01=area1, e02=area2, t02=t02, polar_m1=polar1, polar_m2=polar2, bx=bx, bz=bz, state_param=x0, polarizations_param=p0, delta_B=delta_b, d0=d0, d1=0.12, d2=0.05, delta_E=delta_E)
    # s: g,dp,bp,bm,dm,B
    # t = np.linspace(-t0,t0,len(states))
    return ends, s, t, pol, energy_pulse2
