from detuned_excitation.two_level_system import tls_commons, pulse
import numpy as np
import matplotlib.pyplot as plt

HBAR = 6.582119514e2  # meV fs


def test_rabifreq():
    """
    this is probably not correct yet
    """
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

def fm_pulsed_excitation(tau=10000, dt=4, area=7*np.pi, detuning=-10, small_detuning=3, phase=0, use_t_zero=True, plot=False):
    """
    excites a two level system using a frequency modulated laser pulse.
    tau: width of the gaussian shape laser pulse in femto seconds
    dt: timestep for rk4 integration in fs
    area: pulse area of pulse
    detuning: base detuning of the laser field in meV
    small_detuning: amplitude of the frequency modulation in meV
    phase: adds a phase to the frequency modulation. it seems like this is not relevant though.
    returns: time, array x containing electron occupation (x[:,0]) and polarization (x[:,1]), a pulse object  
    """
    # choose a wide enough time window
    t0 = -4*tau
    t1 = 4*tau
    s = int((t1 - t0) / dt)
    t = np.linspace(t0, t1, s + 1)
    # this just sets the envelope of the laser. setting the frequency happens below.
    p = pulse.Pulse(tau=tau, e_start=0, w_gain=0, e0=area)
    # we now want to set the frequency with something oscillating like the rabi-freq of the system
    # so we first need the time dependent rabi frequency.
    # notice that we set the frequency (!) i.e. the laser energy, not directly the oscillating
    # part of the laser, exp(i*phi(t)). to get phi(t), we would have to
    # integrate the frequency over time.
    # using a rotating frame with the light frequency, we only need the frequencies.
    # this changes if we want a different rotating frame, for example if we want to
    # consider two overlapping pulses.
    detuning_f = detuning/HBAR
    small_det_f = small_detuning/HBAR
    rf = lambda t: np.sqrt((p.get_envelope_f()(t))**2 + detuning_f**2)
    freq = lambda t: detuning_f + small_det_f*np.sin(rf(0)*t+phase)
    if use_t_zero == False:
        freq = lambda t: detuning_f + small_det_f*np.sin(rf(t)*t+phase)
    
    # p.set_frequency(lambda t: 60/(1000**2)*t)  # this would be a chirped excitation like above

    # this one finally sets the frequency(t) using a lambda function
    p.set_frequency(freq)
    x0 = np.array([0,0],dtype=complex)
    _, x = tls_commons.runge_kutta(t0, x0, t1, dt, tls_commons.bloch_eq, p, 0)
    if plot:
        # plt.plot(t,freq(t)*HBAR)
        # plt.show() 
        print("rabifreq(t=0)={:.4f} 1/fs".format(rf(0)))
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        # ax2.plot(t,freq(t)*HBAR, 'b-')
        ax.plot(t,x[:,0].real, 'r-')
        ax.set_ylim(-0.05,1.05)
        ax.set_xlabel("t in fs")
        ax.set_ylabel("Besetzung")
        plt.title("{:.0f}fs, {:.0f}pi, {}meV, {}meV".format(tau,area/np.pi,detuning, small_detuning))
        # ax2.set_ylim(-10,-20)
        plt.show()
    return t, x, p

# fm_pulsed_excitation()
# fm_pulsed_excitation(tau=6000, area=2*np.pi, detuning=-0.1, small_detuning=0)
# fm_pulsed_excitation(tau=6000, area=8*np.pi, detuning=-8, small_detuning=2)
# fm_pulsed_excitation(tau=9000, area=10*np.pi, detuning=-10, small_detuning=2)
# fm_pulsed_excitation(tau=9000, area=7*np.pi, detuning=7, small_detuning=2)
# fm_pulsed_excitation(tau=9000, area=4*np.pi, detuning=3, small_detuning=1.5)

def fm_rect_pulse(tau=10000, dt=4, area=7*np.pi, detuning=-10, small_detuning=3, phase=0):
    """
    excites a two level system using a frequency modulalted rectangle shape pulse.
    see fm_pulsed_excitation() above for more information.  
    """
    t0 = -2*tau
    t1 = 2*tau
    s = int((t1 - t0) / dt)
    t = np.linspace(t0, t1, s + 1)
    p = pulse.RectanglePulse(tau=tau, e_start=0, w_gain=0, e0=area)
   
    detuning = detuning/HBAR  # in rad/fs
    small_det = small_detuning/HBAR  # rad/fs
    rf = lambda t: np.sqrt((p.get_envelope_f()(t))**2 + detuning**2)
    freq = lambda t: detuning + small_det*np.sin(rf(0)*t+phase)
    # print("max. rabifreq: {:.4f} rad/fs or {:.4f} THz (1/ps)".format(rf(0),1000*rf(0)/(2*np.pi)))
    # plt.plot(t,np.array([freq(i) for i in t])*HBAR)
    # plt.show() 
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

# detuned_rect_pulse(tau=20000, area=10*np.pi, detuning=-7, small_detuning=2)

# findings:
# high area leads to more full oscillations of the occupation
# high detuning leads to higher frequency of the small oscillations
# high small_detuning leads to greater leaps per small oscillation (?)


def freq_image():
    t = np.linspace(-1000,1000,200)
    plt.plot(t, -10 + 3 * np.sin(0.0152*t))
    plt.xlabel("t in fs")
    plt.ylabel("Detuning, meV")
    plt.ylim([-15,0])
    plt.show()

# freq_image()