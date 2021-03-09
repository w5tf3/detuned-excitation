import numpy as np
import matplotlib.pyplot as plt
from detuned_excitation.two_level_system import tls_commons, pulse
import tqdm


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

#   beat()

def test_beat(tau1=5000, tau2=5000, dt=5, area1=10*np.pi, area2=10*np.pi, detuning=-5, t02=0, phase=0):
    """
    python version works too, but is slower than fortran
    """
    # take a time window which fits both pulses, even if one is centered around t != 0
    tau = tau1 if tau1 > (tau2+np.abs(t02)) else (tau2+np.abs(t02))
    t0 = -4*tau
    t1 = 4*tau
    s = int((t1 - t0) / dt)
    t = np.linspace(t0, t1, s + 1)
    _t0=0
    # rotating frame with detuning, so e_start = 0
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

    p_total = pulse.MultiPulse(p1, p2)
    x0 = np.array([0,0],dtype=complex)
    _, x = tls_commons.runge_kutta(t0, x0, t1, dt, tls_commons.bloch_eq_constrf, p_total, detuning/HBAR)
    #fig, ax = plt.subplots()
    #ax2 = ax.twinx()
    #ax2.plot(t,freq(t)*HBAR, 'b-')
    #ax.plot(t,x[:,0].real)
    #ax.set_ylim(-0.1,1.1)
    #ax2.set_ylim(-10,-20)
    #plt.show()
    return t, x, p_total

# test_beat(dt=1, tau1=6200, tau2=9600, area1=29.0*np.pi, area2=29.0*np.pi, t02=-1800)

def am_twopulse_excitation(tau1=5000, tau2=5000, dt=5, area1=10*np.pi, area2=10*np.pi, detuning=-5, t02=0, factor=1, detuning2=False):
    """
    two pulses added together, forming a beat. t02 is the time difference between the two pulses.
    pulse 1 is centered around t=0, pulse 2 around t02.
    the energy of the first pulse is given as a parameter (detuning), the energy of the second pulse
    is calculated according to detuning - factor * max_rabi_freq (of pulse 1) * HBAR
    factor = 1 seems optimal. For some factor, we will have resonant excitation, so watch out if this is not wanted.
    """
    # take a time window which fits both pulses, even if one is centered around t != 0
    tau = tau1 if tau1 > (tau2+np.abs(t02)) else (tau2+np.abs(t02))
    t0 = 4*tau
    t = np.arange(-t0,t0,dt)

    # here we calculate the laser frequency for the second pulse
    p1 = pulse.Pulse(tau=tau1, e_start=detuning, w_gain=0, e0=area1, t0=0)
    rf = lambda t: np.sqrt((p1.get_envelope_f()(t))**2 + (detuning/HBAR)**2)
    rf_max = rf(t=0)  # max of rabifreq
    # the energy of the second laser should be the same as the first one, but reduced by rabifrequency*HBAR
    # factor = 1 seems to be optimal
    energy_pulse2 = detuning - factor * HBAR*rf_max
    if detuning2:
        energy_pulse2 = detuning2
    # print("energy1: {:.4f}meV, energy2: {:.4f}meV".format(detuning, energy_pulse2))

    f,polars,states = tls_commons.twopulse(t0=-t0, dt=dt, t_end=t0-dt,area1=area1, area2=area2, tau1=tau1, tau2=tau2, chirp1=0, chirp2=0, energy1=detuning, energy2=energy_pulse2, t02=t02)
    return f, states, t, polars, energy_pulse2

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
           endvals[i,j],_,_,_,_ = am_twopulse_excitation(detuning=detuning, detuning2=detuning2, t02=t02, dt=dt, tau1=x_ax[j], tau2=tau2, area1=y_ax[i], area2=area2)
    
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
           endvals[i,j],_,_,_,_ = am_twopulse_excitation(t02=t02, dt=dt, tau2=x_ax[j], tau1=tau1, area2=y_ax[i], area1=area1)
    
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    print("{}, tau2:{:.4f}, area2:{:.4f}".format(ind,x_ax[ind[1]],y_ax[ind[0]]/np.pi))
    plt.xlabel("tau2")
    plt.ylabel("aera2/pi")
    plt.pcolormesh(x_ax, y_ax/np.pi, endvals, shading='auto')
    plt.plot(x_ax[ind[1]],y_ax[ind[0]]/np.pi, 'r.')
    plt.colorbar()
    plt.show()
    return x_ax, y_ax, endvals

def test_stability_t0(t0_arr, dt=1, tau1=6192, tau2=9583, area1=29.0*np.pi, area2=29.0*np.pi):
    endvals = np.empty([len(t0_arr)])
    for i in tqdm.trange(len(endvals)):
        endvals[i],_,_,_,_ = am_twopulse_excitation(t02=t0_arr[i], dt=dt, tau2=tau2, tau1=tau1, area2=area2, area1=area1)
    
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    print("{}, t0:{:.4f} fs".format(ind,t0_arr[ind]))
    plt.xlabel("t02")
    plt.plot(t0_arr, endvals)
    plt.plot(t0_arr[ind],endvals[ind], 'r.')
    plt.show()
    return endvals

def test_stability_area(ar1, ar2, tau1, tau2, detuning=-5,detuning2=False, t02=0, dt=1):
    """
    test the stability of an excitation with two detuned pulses with respect to
    certain parameters
    """
    x_ax = ar1
    y_ax = ar2
    endvals = np.empty([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           endvals[i,j],_,_,_,_ = am_twopulse_excitation(t02=t02, dt=dt, tau2=tau2, tau1=tau1, area2=y_ax[i], area1=x_ax[j],detuning=detuning, detuning2=detuning2)
    
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    print("{}, area1:{:.4f}, area2:{:.4f}".format(ind,x_ax[ind[1]]/np.pi,y_ax[ind[0]]/np.pi))
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


def test_stability_tau(ar1, ar2, area1, area2, detuning=-5, t02=0, dt=1):
    """
    test the stability of an excitation with two detuned pulses with respect to
    certain parameters
    """
    x_ax = ar1
    y_ax = ar2
    endvals = np.empty([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           endvals[i,j],_,_,_,_ = am_twopulse_excitation(t02=t02, dt=dt, tau2=y_ax[i], tau1=x_ax[j], area2=area2, area1=area1)
    
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
