from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from detuned_excitation.two_level_system import tls_commons, pulse
import tqdm
from detuned_excitation.two_level_system import helper


HBAR = 6.582119514e2  # meV fs

def elec_field(t, amplitude, tau, frequency):
    """
    frequency as angular frequency omega
    """
    return amplitude / (np.sqrt(2*np.pi)*tau) * np.exp(-t**2/(2*tau**2)) * np.exp(-1j*frequency*t)

def beat():
    t = np.arange(-600,600,0.01)
    ef_1 = elec_field(t-100, 1, 100, 0.15)
    ef_2 = elec_field(t+100, 1, 100, 0.1)
    plt.plot(t, ef_1.real, 'r-')
    plt.plot(t, ef_2.real, 'b-')
    # plt.plot(t, np.real(ef_1 + ef_2), 'b-')
    plt.show()
    plt.plot(t, np.real(ef_1 + ef_2), 'b-')
    plt.show()
    f = np.fft.fft(ef_1+ef_2)
    freqs = np.fft.fftfreq(len(ef_1))*2*np.pi
    plt.plot(freqs,f)
    plt.show()

def get_detuning2(tau1, area1, detuning, rectangular=False, factor=1.0):
    p1 = pulse.Pulse(tau=tau1, e_start=0, w_gain=0, e0=area1, t0=0)
    if rectangular:
        p1 = pulse.RectanglePulse(tau=tau1, e_start=0, w_gain=0, e0=area1, t0=0)
    rf = lambda t: np.sqrt((p1.get_envelope_f()(t))**2 + (detuning/HBAR)**2)
    rf_max = rf(t=0)  # max of rabifreq
    energy_pulse2 = detuning - HBAR*rf_max*factor
    return energy_pulse2

def am_twocolor(tau1=5000, tau2=5000, dt=5, area1=10*np.pi, area2=10*np.pi, detuning=-5, t02=0, phase=0, rectangular=False, factor=1.0, factor2=1.0, detuning2=None, rf_energy=None, x0=None):
    """
    python version works too, but is slower than fortran
    rf_energy: set the energy of the rotating frame. default uses the detuning as rotating frame energy,
    so the energy of the first pulse is 'reduced' to zero. this option also seems to deliver
    the smoothest dynamics on the bloch sphere. 
    """
    # take a time window which fits both pulses, even if one is centered around t != 0
    tau = tau1 if tau1 > (tau2+np.abs(t02)) else (tau2+np.abs(t02))
    t0 = -4*tau
    t1 = 4*tau
    s = int((t1 - t0) / dt)
    t = np.linspace(t0, t1, s + 1)
    _t0=0
    if rf_energy is None:
        rf_energy = detuning
    # rotating frame with "detuning", i.e. frequency (energy) of pulse 1 would be e_pulse1 = 0
    e_pulse1 = detuning - rf_energy
    
    p1 = pulse.Pulse(tau=tau1, e_start=e_pulse1, w_gain=0, e0=area1, t0=0)
    if rectangular:
        p1 = pulse.RectanglePulse(tau=tau1, e_start=e_pulse1, w_gain=0, e0=area1, t0=0)
    rf = lambda t: np.sqrt(factor2*(p1.get_envelope_f()(t))**2 + (detuning/HBAR)**2)
    rf_max = rf(t=0)  # max of rabifreq
    # p1.plot(t0,t1, 200)

    energy_pulse2 = detuning - HBAR*rf_max * factor
    if detuning2 is not None:
        energy_pulse2 = detuning2
    
    e_pulse2 = energy_pulse2 - rf_energy
    # due to the rotating frame and time difference, we get an extra phase depending on
    # the full freuency! we just test 1.5 eV
    # phase2 = ((1500+energy_pulse2)/HBAR)*t02
    # uses rotating frame with the first detuning, so substract the first detuning from the second one
    # for the energy of the second pulse.
    p2 = pulse.Pulse(tau=tau2, e_start=e_pulse2, w_gain=0, e0=area2, t0=t02, phase=phase)
    if rectangular:
        p2 = pulse.RectanglePulse(tau=tau2, e_start=e_pulse2, w_gain=0, e0=area2, t0=t02, phase=phase)
    print("energy p1:  {:.4f} mev".format(detuning))
    print("rf_max: {:.4f}".format(rf_max))
    print("energy p2: {:.4f}meV".format(energy_pulse2))
    print("omega1: {:.4f}, omega2: {:.4f}".format(p1.w_start, p2.w_start))

    p_total = pulse.MultiPulse(p1, p2)
    if x0 is None:
        x0 = np.array([0,0],dtype=complex)
    _, x = tls_commons.runge_kutta(t0, x0, t1, dt, tls_commons.bloch_eq_constrf, p_total, rf_energy/HBAR)
    #fig, ax = plt.subplots()
    #ax2 = ax.twinx()
    #ax2.plot(t,freq(t)*HBAR, 'b-')
    #ax.plot(t,x[:,0].real)
    #ax.set_ylim(-0.1,1.1)
    #ax2.set_ylim(-10,-20)
    #plt.show()
    return t, x, p_total

def am_twocolor_fortran(tau1=5000, tau2=5000, dt=5, area1=10*np.pi, area2=10*np.pi, detuning=-5, t02=0, factor=1.0, factor2=1.0, detuning2=None, phase=0, delta_e=0):
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
    
    # print("energy1: {:.4f}meV, energy2: {:.4f}meV".format(detuning, energy_pulse2))

    f,polars,states = tls_commons.twopulse(t0=-t0, dt=dt, t_end=t0-dt,area1=area1, area2=area2, tau1=tau1, tau2=tau2, chirp1=0, chirp2=0, energy1=detuning, energy2=energy_pulse2, t02=t02, phase=phase, delta_e=delta_e)
    t = np.linspace(-t0,t0,len(states))
    return f, states, t, polars, energy_pulse2

def am_twocolor_biexciton(tau1=5000, tau2=5000, dt=5, area1=10*np.pi, area2=10*np.pi, detuning=-5, t02=0, factor=1.0, factor2=1.0, detuning2=None, phase=0, delta_e=0, delta_b=8, alpha=0):
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
    if alpha != 0:
        tau_new = np.sqrt((alpha**2 / tau1**2) + tau1**2 )
        tau = tau_new
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
    
    # print("energy1: {:.4f}meV, energy2: {:.4f}meV".format(detuning, energy_pulse2))

    f,p,states,polars = tls_commons.biex_am_fortran(t0=-t0, dt=dt, t_end=t0-dt,area1=area1, area2=area2, tau1=tau1, tau2=tau2, det1=detuning, det2=energy_pulse2, t02=t02, phase=phase, delta_e=delta_e, delta_b=delta_b, alpha=alpha)
    t = np.linspace(-t0,t0,len(states))
    return f, states, t, polars, energy_pulse2

def am_twocolor_sixls(tau1=5000, tau2=5000, dt=5, area1=10*np.pi, area2=10*np.pi, detuning=-5, t02=0, factor=1.0, factor2=1.0, detuning2=None, polar_m1=1.0, polar_m2=1.0, bx=0, bz=0, state_param=np.array([1, 0, 0, 0, 0, 0]), polarizations_param=np.zeros([15], dtype=complex), delta_b=-0.25, d0=0.25, d1=0.12, d2=0.05, delta_e=0.0):
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
    
    print("energy1: {:.4f}meV, energy2: {:.4f}meV".format(detuning, energy_pulse2))

    # f,p,states,polars = tls_commons.biex_am_fortran(t0=-t0, dt=dt, t_end=t0-dt,area1=area1, area2=area2, tau1=tau1, tau2=tau2, det1=detuning, det2=energy_pulse2, t02=t02, phase=phase, delta_e=delta_e, delta_b=delta_b)
    # t = np.linspace(-t0,t0,len(states))
    E_dp, E_bp, E_bm, E_dm, E_b = helper.energies(bz, delta_b, delta_e, d0)
    detuning += E_bm
    energy_pulse2 += E_bm
    print("energy1: {:.4f}meV, energy2: {:.4f}meV".format(detuning, energy_pulse2))
    t, states, s, p, polars = tls_commons.six_levels_two_color(-t0, t0, dt, tau1, tau2, energy_1=detuning, energy_2=energy_pulse2, e01=area1, e02=area2, t02=t02, polar_m1=polar_m1, polar_m2=polar_m2, bx=bx, bz=bz, state_param=state_param, polarizations_param=polarizations_param, delta_b=delta_b, d0=d0, d1=d1, d2=d2, delta_e=delta_e)
    return s, states, t, polars, energy_pulse2

def am_second_pulse_cw(tau1=5000, dt=5, area1=10*np.pi, area2=10*np.pi, detuning=-5, factor=1.0, factor2=1.0, detuning2=None, rf_energy=None, n_tau=4):
    """
    python version slower than fortran
    rf_energy: set the energy of the rotating frame. default uses the detuning as rotating frame energy,
    so the energy of the first pulse is 'reduced' to zero. this option also seems to deliver
    the smoothest dynamics on the bloch sphere. 
    """
    tau = tau1
    t0 = -n_tau*tau
    t1 = n_tau*tau
    s = int((t1 - t0) / dt)
    t = np.linspace(t0, t1, s + 1)
    _t0=0
    if rf_energy is None:
        rf_energy = detuning
    # rotating frame with "detuning", i.e. frequency (energy) of pulse 1 would be e_pulse1 = 0
    e_pulse1 = detuning - rf_energy
    
    p1 = pulse.Pulse(tau=tau1, e_start=e_pulse1, w_gain=0, e0=area1, t0=0)
    rf = lambda t: np.sqrt(factor2*(p1.get_envelope_f()(t))**2 + (detuning/HBAR)**2)
    rf_max = rf(t=0)  # max of rabifreq

    energy_pulse2 = detuning - HBAR*rf_max * factor
    if detuning2 is not None:
        energy_pulse2 = detuning2
    
    e_pulse2 = energy_pulse2 - rf_energy
    # now, the second pulse is just on as cw during the whole simulation, ie 8*tau
    p2 = pulse.RectanglePulse(tau=2*n_tau*t1, e_start=e_pulse2, w_gain=0, e0=area2, t0=0)
    print("energy p1:  {:.4f} mev".format(detuning))
    print("rf_max: {:.4f}".format(rf_max))
    print("energy p2: {:.4f}meV".format(energy_pulse2))
    print("omega1: {:.4f}, omega2: {:.4f}".format(p1.w_start, p2.w_start))

    p_total = pulse.MultiPulse(p1, p2)
    x0 = np.array([0,0],dtype=complex)
    _, x = tls_commons.runge_kutta(t0, x0, t1, dt, tls_commons.bloch_eq_constrf, p_total, rf_energy/HBAR)
    return t, x, p_total

def am_cw_fortran(tau1=5000, dt=5, area1=10*np.pi, area2=10*np.pi, detuning=-5, factor=1.0, factor2=1.0, detuning2=None, t02=0, n_tau=4):
    """
    the duration of the simulation is 8*tau1: from -4*tau1 to 4*tau1
    """
    # take a time window which fits both pulses, even if one is centered around t != 0
    tau = tau1 
    t0 = n_tau*tau
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
    
    # print("energy1: {:.4f}meV, energy2: {:.4f}meV".format(detuning, energy_pulse2))

    f,polars,states = tls_commons.twopulse_cw(t0=-t0, dt=dt, t_end=t0-dt,area1=area1, area2=area2, tau1=tau1, chirp1=0, energy1=detuning, energy2=energy_pulse2, t02=t02)
    t = np.linspace(-t0,t0,len(states))
    return f, states, t, polars, energy_pulse2

def am_first_cw(tau1=10000, tau2=2000, dt=5, e01=1.0, area2=10*np.pi, detuning=-5, factor=1.0, factor2=1.0, detuning2=None, rf_energy=None, onoff=100):
    """
    python version: first pulse is a smooth (sigmoid on/off) rectangle pulse
    onoff is the swtich time in fs
    e01 is its amplitude in meV, i.e., the amplitude Omega0 = e01/hbar 
    rf_energy: set the energy of the rotating frame. default uses the detuning as rotating frame energy,
    so the energy of the first pulse is 'reduced' to zero. this option also seems to deliver
    the smoothest dynamics on the bloch sphere. 
    """
    # tau = tau2
    t0 = -1.5*tau1/2
    t1 = 1.5*tau1/2
    s = int((t1 - t0) / dt)
    t = np.linspace(t0, t1, s + 1)
    _t0=0
    if rf_energy is None:
        rf_energy = detuning
    # rotating frame with "detuning", i.e. frequency (energy) of pulse 1 would be e_pulse1 = 0
    e_pulse1 = detuning - rf_energy
    
    p1 = pulse.SmoothRectangle(tau=tau1, e_start=e_pulse1, w_gain=0, e0=e01/HBAR, t0=0, alpha_onoff=onoff)
    rf_max = np.sqrt(factor2*(e01/HBAR)**2 + (detuning/HBAR)**2)  # rabi freq
    
    energy_pulse2 = detuning - HBAR*rf_max * factor
    if detuning2 is not None:
        energy_pulse2 = detuning2
    
    e_pulse2 = energy_pulse2 - rf_energy
    # now, the second pulse is just on as cw during the whole simulation, ie 8*tau
    p2 = pulse.Pulse(tau=tau2, e_start=e_pulse2, w_gain=0, e0=area2, t0=0)
    print("energy p1:  {:.4f} mev".format(detuning))
    print("hbar*rf_max: {:.4f}".format(rf_max*HBAR))
    print("energy p2: {:.4f}meV".format(energy_pulse2))
    # print("omega1: {:.4f}, omega2: {:.4f}".format(p1.w_start, p2.w_start))

    p_total = pulse.MultiPulse(p1, p2)
    x0 = np.array([0,0],dtype=complex)
    _, x = tls_commons.runge_kutta(t0, x0, t1, dt, tls_commons.bloch_eq_constrf, p_total, rf_energy/HBAR)
    return t, x, p_total

def test_beat_special_frame(tau1=5000, tau2=5000, dt=5, area1=10*np.pi, area2=10*np.pi, detuning=-5, t02=0, phase=0):
    # take a time window which fits both pulses, even if one is centered around t != 0
    tau = tau1 if tau1 > (tau2+np.abs(t02)) else (tau2+np.abs(t02))
    t0 = -4*tau
    t1 = 4*tau
    s = int((t1 - t0) / dt)
    t = np.linspace(t0, t1, s + 1)
    _t0=0
    # rotating frame with "detuning", i.e. frequency (energy) of pulse 1, so e_start = 0
    p1 = pulse.Pulse(tau=tau1, e_start=0, w_gain=0, e0=area1, t0=0)
    rf = lambda t: np.sqrt((p1.get_envelope_f()(t))**2 + (detuning/HBAR)**2)
    rf_max = rf(t=0)  # max of rabifreq
    # p1.plot(t0,t1, 200)

    energy_pulse2 = detuning - HBAR*rf_max

    # due to the rotating frame and time difference, we get an extra phase depending on
    # the full freuency! we just test 1.5 eV
    # phase2 = ((1500+energy_pulse2)/HBAR)*t02
    # uses rotating frame with the first detuning, so substract the first detuning from the second one
    # for the energy of the second pulse.
    p2 = pulse.Pulse(tau=tau2, e_start=energy_pulse2-detuning, w_gain=0, e0=area2, t0=t02, phase=phase)

    print("energy p1:  {:.4f} mev".format(detuning))
    print("rf_max: {:.4f}".format(rf_max))
    print("energy p2: {:.4f}meV".format(energy_pulse2))
    print("omega1: {:.4f}, omega2: {:.4f}".format(p1.w_start, p2.w_start))

    freq1 = detuning/HBAR
    freq2 = energy_pulse2/HBAR
    delta = freq2 - freq1 
    omega1 = p1.get_envelope_f()
    omega2 = p2.get_envelope_f()

    # basically freq. for first pulse
    delta_1 = lambda t: (omega2(t)**2 - omega1(t)**2 + delta**2)/(2*delta)
    # delta_1 = lambda t: 0
    # phase for first pulse
    phase1 = lambda t: -t*delta_1(t)
    # phase for second pulse
    phase2 = lambda t: -t*(freq1 - freq2 + delta_1(t))

    p1_ = pulse.Pulse(tau=tau1, e_start=0, w_gain=0, e0=area1, t0=0)
    p1_.set_phase(phase1)
    p2_ = pulse.Pulse(tau=tau2, e_start=0, w_gain=0, e0=area2, t0=t02, phase=phase)
    p2_.set_phase(phase2)

    freq_rf = lambda t : freq1 + delta_1(t) + t * 1/(2*delta) * (p1_.get_envelope(t)*p1_.get_envelope_derivative(t) - 
                                      p2_.get_envelope(t)*p2_.get_envelope_derivative(t))
    p_total = pulse.MultiPulse(p1_, p2_, freq_rf)    
    x0 = np.array([0,0],dtype=complex)
    _, x = tls_commons.runge_kutta(t0, x0, t1, dt, tls_commons.bloch_eq_general_rf, p_total, 0)
    #fig, ax = plt.subplots()
    #ax2 = ax.twinx()
    #ax2.plot(t,freq(t)*HBAR, 'b-')
    #ax.plot(t,x[:,0].real)
    #ax.set_ylim(-0.1,1.1)
    #ax2.set_ylim(-10,-20)
    #plt.show()
    return t, x, p_total

def test_beat_special_frame1(tau1=5000, tau2=5000, dt=5, area1=10*np.pi, area2=10*np.pi, detuning=-5, t02=0, phase=0, detuning2=None):
    # take a time window which fits both pulses, even if one is centered around t != 0
    tau = tau1 if tau1 > (tau2+np.abs(t02)) else (tau2+np.abs(t02))
    t0 = -4*tau
    t1 = 4*tau
    s = int((t1 - t0) / dt)
    t = np.linspace(t0, t1, s + 1)
    _t0=0
    # rotating frame with "detuning", i.e. frequency (energy) of pulse 1, so e_start = 0
    p1 = pulse.Pulse(tau=tau1, e_start=0, w_gain=0, e0=area1, t0=0)
    rf = lambda t: np.sqrt((p1.get_envelope_f()(t))**2 + (detuning/HBAR)**2)
    rf_max = rf(t=0)  # max of rabifreq
    # p1.plot(t0,t1, 200)

    energy_pulse2 = detuning - HBAR*rf_max
    if detuning2 is not None:
        energy_pulse2 = detuning2

    # due to the rotating frame and time difference, we get an extra phase depending on
    # the full freuency! we just test 1.5 eV
    # phase2 = ((1500+energy_pulse2)/HBAR)*t02
    # uses rotating frame with the first detuning, so substract the first detuning from the second one
    # for the energy of the second pulse.
    p2 = pulse.Pulse(tau=tau2, e_start=energy_pulse2-detuning, w_gain=0, e0=area2, t0=t02, phase=phase)

    print("energy p1:  {:.4f} mev".format(detuning))
    print("rf_max: {:.4f}".format(rf_max))
    print("energy p2: {:.4f}meV".format(energy_pulse2))
    print("omega1: {:.4f}, omega2: {:.4f}".format(p1.w_start, p2.w_start))

    freq1 = detuning/HBAR
    freq2 = energy_pulse2/HBAR
    delta = freq2 - freq1 
    omega1 = p1.get_envelope_f()
    omega2 = p2.get_envelope_f()

    # basically freq. for first pulse
    delta_1 = (omega2(t02)**2 - omega1(0)**2 + delta**2)/(2*delta)
    #delta_1=0
    # delta_1 = (omega2(t02)**2 - omega1(0)**2 + delta**2)/(2*delta)
    # phase for first pulse
    # phase1 = lambda t: t*delta_1(t)
    # phase for second pulse
    # phase2 = lambda t: t*(freq1 - freq2 + delta_1(t))

    p1_ = pulse.Pulse(tau=tau1, e_start=-delta_1*HBAR, w_gain=0, e0=area1, t0=0)
    p2_ = pulse.Pulse(tau=tau2, e_start=-(freq1 - freq2 + delta_1)*HBAR, w_gain=0, e0=area2, t0=t02, phase=phase)

    p_total = pulse.MultiPulse(p1_, p2_)    
    x0 = np.array([0,0],dtype=complex)
    _, x = tls_commons.runge_kutta(t0, x0, t1, dt, tls_commons.bloch_eq_constrf, p_total, freq1+delta_1)
    #fig, ax = plt.subplots()
    #ax2 = ax.twinx()
    #ax2.plot(t,freq(t)*HBAR, 'b-')
    #ax.plot(t,x[:,0].real)
    #ax.set_ylim(-0.1,1.1)
    #ax2.set_ylim(-10,-20)
    #plt.show()
    return t, x, p_total

def test_beat_special_frame2(tau1=5000, tau2=5000, dt=5, area1=10*np.pi, area2=10*np.pi, detuning=-5, t02=0, phase=0):
    # take a time window which fits both pulses, even if one is centered around t != 0
    tau = tau1 if tau1 > (tau2+np.abs(t02)) else (tau2+np.abs(t02))
    t0 = -4*tau
    t1 = 4*tau
    s = int((t1 - t0) / dt)
    t = np.linspace(t0, t1, s + 1)
    _t0=0
    # rotating frame with "detuning", i.e. frequency (energy) of pulse 1, so e_start = 0
    p1 = pulse.Pulse(tau=tau1, e_start=0, w_gain=0, e0=area1, t0=0)
    rf = lambda t: np.sqrt((p1.get_envelope_f()(t))**2 + (detuning/HBAR)**2)
    rf_max = rf(t=0)  # max of rabifreq
    # p1.plot(t0,t1, 200)

    energy_pulse2 = detuning - HBAR*rf_max

    # due to the rotating frame and time difference, we get an extra phase depending on
    # the full freuency! we just test 1.5 eV
    # phase2 = ((1500+energy_pulse2)/HBAR)*t02
    # uses rotating frame with the first detuning, so substract the first detuning from the second one
    # for the energy of the second pulse.
    p2 = pulse.Pulse(tau=tau2, e_start=energy_pulse2-detuning, w_gain=0, e0=area2, t0=t02, phase=phase)

    print("energy p1:  {:.4f} mev".format(detuning))
    print("rf_max: {:.4f}".format(rf_max))
    print("energy p2: {:.4f}meV".format(energy_pulse2))
    print("omega1: {:.4f}, omega2: {:.4f}".format(p1.w_start, p2.w_start))

    freq1 = detuning/HBAR
    freq2 = energy_pulse2/HBAR
    delta = freq2 - freq1 
    omega1 = p1.get_envelope_f()
    omega2 = p2.get_envelope_f()

    # basically freq. for first pulse
    delta_1 = lambda t: (omega2(t)**2 - omega1(t)**2 + delta**2)/(2*delta)
    # phase for first pulse
    phase1 = lambda t: 0.5*delta*t + 1/(2*delta)*(p2.get_envelope_square_integral(t)-p1.get_envelope_square_integral(t))
    # phase for second pulse
    phase2 = lambda t: t*(freq1-freq2) + phase1(t)

    p1_ = pulse.Pulse(tau=tau1, e_start=0, w_gain=0, e0=area1, t0=0)
    p1_.set_phase(phase1)
    p2_ = pulse.Pulse(tau=tau2, e_start=0, w_gain=0, e0=area2, t0=t02, phase=phase)
    p2_.set_phase(phase2)

    freq_rf = lambda t : freq1 + delta_1(t)
    p_total = pulse.MultiPulse(p1_, p2_, freq_rf)    
    x0 = np.array([0,0],dtype=complex)
    _, x = tls_commons.runge_kutta(t0, x0, t1, dt, tls_commons.bloch_eq_general_rf, p_total, 0)
    #fig, ax = plt.subplots()
    #ax2 = ax.twinx()
    #ax2.plot(t,freq(t)*HBAR, 'b-')
    #ax.plot(t,x[:,0].real)
    #ax.set_ylim(-0.1,1.1)
    #ax2.set_ylim(-10,-20)
    #plt.show()
    return t, x, p_total
# test_beat(dt=1, tau1=6200, tau2=9600, area1=29.0*np.pi, area2=29.0*np.pi, t02=-1800)

def test_stability_pulse1(dur1, ar1, tau2, area2, detuning=-5, t02=0, dt=1, detuning2=False):
    """
    test the stability of an excitation with two detuned pulses with respect to
    certain parameters
    """
    x_ax = dur1
    y_ax = ar1
    endvals = np.empty([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           endvals[i,j],_,_,_,_ = am_twocolor_fortran(detuning=detuning, detuning2=detuning2, t02=t02, dt=dt, tau1=x_ax[j], tau2=tau2, area1=y_ax[i], area2=area2)
    
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    print("{}, tau1:{:.4f}, area1:{:.4f}".format(ind,x_ax[ind[1]],y_ax[ind[0]]/np.pi))
    plt.xlabel("tau1")
    plt.ylabel("aera1/pi")
    plt.pcolormesh(x_ax, y_ax/np.pi, endvals, shading='auto')
    plt.plot(x_ax[ind[1]],y_ax[ind[0]]/np.pi, 'r.')
    plt.colorbar()
    plt.show()
    return x_ax, y_ax, endvals

def test_stability_pulse2(dur2, ar2, tau1, area1, detuning=-5, t02=0, dt=1):
    """
    test the stability of an excitation with two detuned pulses with respect to
    certain parameters
    """
    x_ax = dur2
    y_ax = ar2
    endvals = np.empty([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           endvals[i,j],_,_,_,_ = am_twocolor_fortran(t02=t02, dt=dt, tau2=x_ax[j], tau1=tau1, area2=y_ax[i], area1=area1)
    
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    print("{}, tau2:{:.4f}, area2:{:.4f}".format(ind,x_ax[ind[1]],y_ax[ind[0]]/np.pi))
    plt.xlabel("tau2")
    plt.ylabel("aera2/pi")
    plt.pcolormesh(x_ax, y_ax/np.pi, endvals, shading='auto')
    plt.plot(x_ax[ind[1]],y_ax[ind[0]]/np.pi, 'r.')
    plt.colorbar()
    plt.show()
    return x_ax, y_ax, endvals

def biexciton_stability_pulse2(tau1, tau2, area1, area2s, detuning, detuning2s, t02=0, dt=5, delta_b=4):
    """
    test the stability of an excitation with two detuned pulses with respect to
    certain parameters
    """
    x_ax = detuning2s
    y_ax = area2s
    endvals = np.empty([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           _,s,_,_,_ = am_twocolor_biexciton(tau1=tau1, tau2=tau2, area1=area1, area2=y_ax[i], t02=t02, detuning=detuning, detuning2=x_ax[j], delta_b=delta_b)
           endvals[i,j] = s[-1,1]
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    print("{}, detuning2:{:.4f}, area2:{:.4f}, endval:{:.4f}".format(ind,x_ax[ind[1]],y_ax[ind[0]]/np.pi,endvals[ind[0],ind[1]]))
    plt.xlabel("detuning2")
    plt.ylabel("area2/pi")
    plt.pcolormesh(x_ax, y_ax/np.pi, endvals, shading='auto')
    plt.plot(x_ax[ind[1]],y_ax[ind[0]]/np.pi, 'r.')
    plt.colorbar(label="X occupation")
    plt.show()
    return x_ax, y_ax, endvals

def test_stability_t0(t0_arr, dt=1, tau1=6192, tau2=9583, area1=29.0*np.pi, area2=29.0*np.pi):
    endvals = np.empty([len(t0_arr)])
    for i in tqdm.trange(len(endvals)):
        endvals[i],_,_,_,_ = am_twocolor_fortran(t02=t0_arr[i], dt=dt, tau2=tau2, tau1=tau1, area2=area2, area1=area1)
    
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    print("{}, t0:{:.4f} fs".format(ind,t0_arr[ind]))
    plt.xlabel("t02")
    plt.plot(t0_arr, endvals)
    plt.plot(t0_arr[ind],endvals[ind], 'r.')
    plt.show()
    return endvals

def test_stability_area(ar1, ar2, tau1, tau2, detuning=-5,detuning2=None, t02=0, dt=1):
    """
    test the stability of an excitation with two detuned pulses with respect to
    certain parameters
    """
    x_ax = ar1
    y_ax = ar2
    endvals = np.empty([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           endvals[i,j],_,_,_,_ = am_twocolor_fortran(t02=t02, dt=dt, tau2=tau2, tau1=tau1, area2=y_ax[i], area1=x_ax[j],detuning=detuning, detuning2=detuning2)
    
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    print("{}, area1:{:.4f}, area2:{:.4f}, final occ.:{:.4f}".format(ind,x_ax[ind[1]]/np.pi,y_ax[ind[0]]/np.pi,endvals[ind[0],ind[1]]))
    plt.xlabel("area1/pi")
    plt.ylabel("area2/pi")
    plt.pcolormesh(x_ax/np.pi, y_ax/np.pi, endvals, shading='auto')
    plt.plot(x_ax[ind[1]]/np.pi,y_ax[ind[0]]/np.pi, 'r.')
    plt.colorbar()
    plt.show()
    return x_ax, y_ax, endvals

def detuning_from_pulse1(ar1, tau1, detuning1=-5):
    x_ax = tau1
    y_ax = ar1
    endvals = np.empty([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
            p1 = pulse.Pulse(tau=x_ax[j], e_start=detuning1, w_gain=0, e0=y_ax[i], t0=0)
            rf = lambda t: np.sqrt((p1.get_envelope_f()(t))**2 + (detuning1/HBAR)**2)
            rf_max = rf(t=0)
            endvals[i,j] = - (detuning1 - HBAR*rf_max)
      # max of rabifreq
    # the energy of the second laser should be the same as the first one, but reduced by rabifrequency*HBAR
    # factor = 1 seems to be optimal
    #ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    #max_x,max_y = x_ax[ind[1]],y_ax[ind[0]]
    #print("{}, tau1:{:.4f}, tau2:{:.4f}, endval:{:.4f}".format(ind,max_x,max_y,endvals[ind[0],ind[1]]))
    plt.xlabel("tau1")
    plt.ylabel("area1/pi")
    plt.pcolormesh(x_ax, y_ax/np.pi, endvals, shading='auto')
    #plt.plot(x_ax[ind[1]],y_ax[ind[0]], 'r.')
    plt.colorbar()
    plt.show()
    return x_ax, y_ax, endvals

def test_stability_tau(tau1s, tau2s, area1, area2, detuning=-5, t02=0, dt=1):
    """
    test the stability of an excitation with two detuned pulses with respect to
    certain parameters
    """
    x_ax = tau1s
    y_ax = tau2s
    endvals = np.empty([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           endvals[i,j],_,_,_,_ = am_twocolor_fortran(t02=t02, dt=dt, tau2=y_ax[i], tau1=x_ax[j], area2=area2, area1=area1)
    
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    max_x,max_y = x_ax[ind[1]],y_ax[ind[0]]
    print("{}, tau1:{:.4f}, tau2:{:.4f}, endval:{:.4f}".format(ind,max_x,max_y,endvals[ind[0],ind[1]]))
    plt.xlabel("tau1")
    plt.ylabel("tau2")
    plt.pcolormesh(x_ax, y_ax, endvals, shading='auto')
    plt.plot(x_ax[ind[1]],y_ax[ind[0]], 'r.')
    plt.colorbar()
    plt.show()
    return x_ax, y_ax, endvals

def stability_detuning2_tau02(det2s, tau02s, tau1, tau2, area1, area2, detuning, dt=1):
    x_ax = det2s
    y_ax = tau02s
    endvals = np.empty([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           endvals[i,j],_,_,_,_ = am_twocolor_fortran(t02=y_ax[i], detuning2=x_ax[j], dt=dt, tau2=tau2, tau1=tau1, area2=area2, area1=area1, detuning=detuning)
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    max_x,max_y = x_ax[ind[1]],y_ax[ind[0]]
    print("{}, det2:{:.4f}, t02:{:.4f}, endval:{:.4f}".format(ind,max_x,max_y,endvals[ind[0],ind[1]]))
    plt.xlabel("det2")
    plt.ylabel("t02")
    plt.pcolormesh(x_ax, y_ax, endvals, shading='auto')
    plt.plot(x_ax[ind[1]],y_ax[ind[0]], 'r.')
    plt.colorbar()
    plt.show()
    return x_ax, y_ax, endvals

def search_optimum(tau1, tau2, dt, area1, area2, detuning, t02, n=10, percent=2, detuning2=False):
    low = (100-percent)/100
    high = (100+percent)/100
    tau1s = np.linspace(tau1*low,tau1*high,n)
    tau2s = np.linspace(tau2*low,tau2*high,n)
    area1s = np.linspace(area1*low,area1*high,n)
    area2s = np.linspace(area2*low,area2*high,n)
    endvals = np.empty([n,n,n,n])
    for v1 in tqdm.trange(n):
        for v2 in range(n):
            for v3 in range(n):
                for v4 in range(n):
                    endvals[v1,v2,v3,v4],_,_,_,_ = am_twocolor_fortran(dt=dt,tau1=tau1s[v1],tau2=tau2s[v2],area1=area1s[v3],area2=area2s[v4],t02=t02,detuning=detuning,detuning2=detuning2)
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    v1,v2,v3,v4 = ind[0],ind[1],ind[2],ind[3]
    m_tau1 = tau1s[v1]
    m_tau2 = tau2s[v2]
    m_area1 = area1s[v3]
    m_area2 = area2s[v4]
    print("endavlue: {:.4f}".format(endvals[v1,v2,v3,v4]))
    print("tau1={:.3f},tau2={:.3f},area1={:.3f}*np.pi,area2={:.3f}*np.pi".format(m_tau1,m_tau2,m_area1/np.pi,m_area2/np.pi))

def four_parameter_stability(taus, areas, tau2s, area2s, detuning1, t02s, dt=4):
    x_ax = taus
    y_ax = areas
    # area2s = areas
    # tau2s = taus + delta_tau
    endvals = np.empty([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           endvals[i,j],_,_,_,_ = am_twocolor_fortran(tau2=tau2s[j], tau1=x_ax[j], area2=area2s[i], area1=y_ax[i], t02=t02s[j], detuning=detuning1, dt=dt)
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    max_x,max_y = x_ax[ind[1]],y_ax[ind[0]]
    print("{}, tau1={:.4f}, area1={:.4f}*np.pi, area2={:.4f}*np.pi, tau2={:.4f}, t02={:.4f}, endval:{:.4f}".format(ind,max_x,max_y/np.pi,area2s[ind[0]]/np.pi, tau2s[ind[1]], t02s[ind[1]], endvals[ind[0],ind[1]]))
    plt.xlabel("tau1")
    plt.ylabel("areas/np.pi")
    plt.pcolormesh(x_ax, y_ax/np.pi, endvals, shading='auto')
    plt.plot(x_ax[ind[1]],y_ax[ind[0]]/np.pi, 'r.')
    # plt.plot(x_ax, 12*np.sqrt(x_ax/1000))
    # m = max_y / (max_x *np.pi)
    # plt.plot(x_ax, m * x_ax, 'r-')
    plt.colorbar()
    plt.show()
    return x_ax, y_ax, endvals

def stability_factor2(factors=np.linspace(-1,2,200), tau1=2400, tau2=3000, area1=22.5*np.pi, area2=20.0*np.pi, t02=-700, detuning=-8.0000):
    endvals = []
    energies = []
    for v in factors:
        f, _, _, _, e2 = am_twocolor_fortran(factor2=v, tau1=2400, tau2=3000, area1=22.5*np.pi, area2=20.0*np.pi, t02=-700, detuning=-8.0000)
        endvals.append(f)
        energies.append(e2)
    plt.xlabel("factor")
    plt.ylabel("final occupation")
    plt.plot(factors,endvals)
    plt.show()

    plt.xlabel("factor")
    plt.ylabel("energy of second pulse (meV)")
    plt.plot(factors,energies)
    plt.show()

    plt.xlabel("energy of second pulse (meV)")
    plt.ylabel("final occupation")
    plt.plot(energies, endvals)
    plt.show()

def cw_detuning_area(dets, areas_cw, tau1, area1, t02=0, dt=1):
    x_ax = dets
    y_ax = areas_cw/(8*tau1)  # the simulation length is 8*tau
    endvals = np.empty([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           endvals[i,j],_,_,_,_ = am_cw_fortran(tau1=tau1, area1=area1, area2=y_ax[i], detuning=x_ax[j], t02=t02)
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    max_x,max_y = x_ax[ind[1]],y_ax[ind[0]]
    print("{}, det1:{:.4f}, area_cw:{:.4f}, endval:{:.4f}".format(ind,max_x,max_y*(8*tau1/np.pi),endvals[ind[0],ind[1]]))
    plt.xlabel("detuning1")
    plt.ylabel("area_cw/pi")
    plt.pcolormesh(x_ax, y_ax*(8*tau1/np.pi), endvals, shading='auto')
    plt.plot(x_ax[ind[1]],y_ax[ind[0]]*(8*tau1/np.pi), 'r.')
    plt.colorbar()
    plt.show()
    return x_ax, y_ax, endvals

def detuning_area(det2s, area2s, det1, tau1, tau2, area1, t02=0, dt=1):
    x_ax = det2s
    y_ax = area2s
    endvals = np.empty([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           endvals[i,j],_,_,_,_ = am_twocolor_fortran(dt=dt, detuning=det1, tau1=tau1, tau2=tau2, area1=area1, area2=y_ax[i], detuning2=x_ax[j], t02=t02)
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    max_x,max_y = x_ax[ind[1]],y_ax[ind[0]]
    print("{}, det2:{:.4f}, area2:{:.4f}, endval:{:.4f}".format(ind,max_x,max_y/np.pi,endvals[ind[0],ind[1]]))
    plt.xlabel("detuning2")
    plt.ylabel("area2/pi")
    plt.pcolormesh(x_ax, y_ax/np.pi, endvals, shading='auto')
    plt.plot(x_ax[ind[1]],y_ax[ind[0]]/np.pi, 'r.')
    #plt.plot(x_ax, (1/np.pi)*np.sqrt(2*np.pi*tau1**2)*(1/HBAR)*np.sqrt((det1- x_ax)**2-det1**2), 'r-')
    plt.colorbar()
    plt.show()
    return x_ax, y_ax, endvals
