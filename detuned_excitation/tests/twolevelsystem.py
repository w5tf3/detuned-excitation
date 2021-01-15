from detuned_excitation.two_level_system import tls_commons, pulse
import numpy as np
import matplotlib.pyplot as plt

HBAR = 6.582119514e2  # meV fs


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


def test_python():
    p = pulse.Pulse(tau=400, e_start=0, w_gain=60/(1000**2), e0=8*np.pi)
    print(p.get_energies())
    x0 = np.array([0,0],dtype=complex)
    t, x2 = tls_commons.runge_kutta(-4*400, x0, 4*400, 2, tls_commons.bloch_eq, p, 0)
    plt.plot(t,x2[:,0].real)
    plt.show()


def test_rabifreq():
    tau = 500
    area = 8*np.pi
    dt = 10
    dt_small = 0.1
    t = np.arange(-4*tau, 4*tau, dt)
    f = np.zeros([len(t)])
    f_ = 0
    p_ = 0+0j
    ef = tls_commons.Electric_field_envelope(8*np.pi, 400)
    detuning = -15  # meV
    rabifreq = ef.e_f(t)  # momentaneous rabi freq without detuning
    rabifreq = np.sqrt(rabifreq**2 + (detuning/HBAR)**2)
    ef_integrated = np.empty_like(rabifreq)
    ef_integrated[0] = dt*rabifreq[0]
    for i in range(len(ef_integrated)-1):
        ef_integrated[i+1] += rabifreq[i]*dt
    
    modulation = np.sin(rabifreq*t+ef_integrated)
    energies = -15*np.ones_like(t) + 3*modulation
    plt.plot(t, modulation)
    plt.plot(t,energies)
    plt.show()
    # careful: if chirp*t_ is given as frequency, this has to be multiplied by a factor of 0.5
    # because the resulting frequency for some time t_ has to read w_0 + 0.5 * a * t
    for i in range(len(t)-1):
        t_ = t[i]
        
        f_, p_ = tls_commons.two_level(t_,dt_small,t_+dt,f_,p_,energies[i],area,tau)
        f[i+1] = f_
    plt.plot(t,f)
    plt.show()

# test_chirp_a()
#test_chirp()
#test_python()
#test_rabifreq()

def test_excitation(tau=10000, dt=4, area=7*np.pi, detuning=-10, small_detuning=3, phase=0):
    # tau = 10000
    t0 = -4*tau
    t1 = 4*tau
    #dt = 4
    s = int((t1 - t0) / dt)
    t = np.linspace(t0, t1, s + 1)
    # area = 7*np.pi
    p = pulse.Pulse(tau=tau, e_start=0, w_gain=0, e0=area)
    # we now want to set the frequency with something oscillating like the rabi-freq of the system
    # so we first need the time dependent rabi freq
    # notice that we set the frequency (!) i.e. the laser energy, not directly the oscillating
    # part of the laser, exp(i*phi(t)). to get phi(t), we would have to
    # integrate the frequency over time.
    # using a rotating frame with the light frequency, we only need the frequencies.
    # this changes if we want a different rotating frame, for example if we want to
    # consider two overlapping pulses.
    detuning = detuning/HBAR
    small_det = small_detuning/HBAR
    rf = lambda t: np.sqrt((p.get_envelope_f()(t))**2 + detuning**2)
    freq = lambda t: detuning + small_det*np.sin(rf(t)*t+phase)
    # plt.plot(t,freq(t)*HBAR)
    # plt.show() 
    # p.set_frequency(lambda t: 60/(1000**2)*t)  # this would be a chirped excitation like above
    p.set_frequency(freq)
    x0 = np.array([0,0],dtype=complex)
    _, x = tls_commons.runge_kutta(t0, x0, t1, dt, tls_commons.bloch_eq, p, 0)
    # fig, ax = plt.subplots()
    # ax2 = ax.twinx()
    # #ax2.plot(t,freq(t)*HBAR, 'b-')
    # ax.plot(t,x[:,0].real, 'r-')
    # ax.set_ylim(0,1)
    # #ax2.set_ylim(-10,-20)
    # plt.show()
    return t, x

#test_excitation()
#test_excitation(tau=6000, area=8*np.pi, detuning=-8, small_detuning=2)
#test_excitation(tau=9000, area=10*np.pi, detuning=-10, small_detuning=2)
#test_excitation(tau=9000, area=7*np.pi, detuning=7, small_detuning=2)
#test_excitation(tau=9000, area=4*np.pi, detuning=3, small_detuning=1.5)

def detuned_rect_pulse(tau=10000, dt=4, area=7*np.pi, detuning=-10, small_detuning=3, phase=0):
    # tau = 10000
    t0 = -2*tau
    t1 = 2*tau
    #dt = 4
    s = int((t1 - t0) / dt)
    t = np.linspace(t0, t1, s + 1)
    # area = 7*np.pi
    p = pulse.RectanglePulse(tau=tau, e_start=0, w_gain=0, e0=area)
    # p.plot(t0,t1,300)
    # we now want to set the frequency with something oscillating like the rabi-freq of the system
    # so we first need the time dependent rabi freq
    # notice that we set the frequency (!) i.e. the laser energy, not directly the oscillating
    # part of the laser, exp(i*phi(t)). to get phi(t), we would have to
    # integrate the frequency over time.
    # using a rotating frame with the light frequency, we only need the frequencies.
    # this changes if we want a different rotating frame, for example if we want to
    # consider two overlapping pulses.
    detuning = detuning/HBAR
    small_det = small_detuning/HBAR
    rf = lambda t: np.sqrt((p.get_envelope_f()(t))**2 + detuning**2)
    freq = lambda t: detuning + small_det*np.sin(rf(0)*t+phase)
    plt.plot(t,np.array([freq(i) for i in t])*HBAR)
    plt.show() 
    p.set_frequency(freq)
    x0 = np.array([0,0],dtype=complex)
    _, x = tls_commons.runge_kutta(t0, x0, t1, dt, tls_commons.bloch_eq, p, 0)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    #ax2.plot(t,freq(t)*HBAR, 'b-')
    ax.plot(t,[1 for i in t], 'g-')
    ax.plot(t,x[:,0].real, 'r-')
    ax.set_ylim(-0.1,1.1)
    #ax2.set_ylim(-10,-20)
    plt.show()
    return t, x

detuned_rect_pulse(tau=20000, area=10*np.pi, detuning=-10, small_detuning=2)

# findings:
# high area leads to more full oscillations of the occupation
# high detuning leads to higher frequency of the small oscillations
# high small_detuning leads to greater leaps per small oscillation (?)