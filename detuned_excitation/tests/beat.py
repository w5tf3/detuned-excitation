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
    ef_1 = elec_field(t, 1, 100, 0.5)
    ef_2 = elec_field(t, 1, 100, 0.1)
    plt.plot(t, ef_1.real, 'r-')
    plt.plot(t, ef_2.real, 'r-')
    plt.plot(t, np.real(ef_1 + ef_2), 'b-')
    plt.show()
    f = np.fft.fft(ef_1+ef_2)
    freqs = np.fft.fftfreq(len(ef_1))*2*np.pi
    plt.plot(freqs,f)
    plt.show()



# beat()

def test_beat(tau=10000, dt=4, area=7*np.pi, detuning=-3, w_gain=0):
    # tau = 10000
    t0 = -4*tau
    t1 = 4*tau
    #dt = 4
    s = int((t1 - t0) / dt)
    t = np.linspace(t0, t1, s + 1)
    _t0=0
    # area = 7*np.pi
    p1 = pulse.Pulse(tau=tau, e_start=detuning, w_gain=w_gain, e0=area, t0=_t0)
    p1.plot(t0,t1, 200)
    
    rf = lambda t: np.sqrt((p1.get_envelope_f()(t))**2 + (detuning/HBAR)**2)
    rf_max = rf(_t0)
    energy_pulse2 = detuning - 2 * HBAR*rf_max
    p2 = pulse.Pulse(tau=tau, e_start=energy_pulse2, w_gain=0, e0=0*area, t0=0)

    print("energy p1:  {:.4f} mev".format(p1.e_start))
    print("rf_max: {:.4f}".format(rf_max))
    print("energy p2: {:.4f}meV".format(energy_pulse2))
    print("omega1: {:.4f}, omega2: {:.4f}".format(p1.w_start, p2.w_start))

    p_total = pulse.MultiPulse(p1, p2)
    x0 = np.array([0,0],dtype=complex)
    _, x = tls_commons.runge_kutta(t0, x0, t1, dt, tls_commons.bloch_eq_constrf, p1, 0)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    #ax2.plot(t,freq(t)*HBAR, 'b-')
    ax.plot(t,x[:,1].real, 'r-')
    ax.set_ylim(-0.1,1.1)
    #ax2.set_ylim(-10,-20)
    plt.show()
    return t, x

#test_beat(tau=400, detuning=0, w_gain=60/(1000**2))

def test_twopulse(tau1=5000, tau2=5000, dt=5, area1=10*np.pi, area2=10*np.pi, detuning=-5, t02=0, factor=1):
    """
    two pulses added together, forming a beat. t02 is the time difference between the two pulses.
    pulse 1 is centered around t=0, pulse 2 around t02.
    the energy of the first pulse is given as a parameter (detuning), the energy of the second pulse
    is calculated according to detuning - factor * max_rabi_freq (of pulse 1) * HBAR
    factor = 1 seems optimal.
    """
    tau = tau1 if tau1 > (tau2+np.abs(t02)) else (tau2+np.abs(t02))
    t0 = 4*tau
    #dt = 10
    t = np.arange(-t0,t0,dt)

    # here we calculate the right laser frequency for the second pulse
    p1 = pulse.Pulse(tau=tau1, e_start=detuning, w_gain=0, e0=area1, t0=0)
    rf = lambda t: np.sqrt((p1.get_envelope_f()(t))**2 + (detuning/HBAR)**2)
    rf_max = rf(t=0)  # max of rabifreq
    # the energy of the second laser should be the same as the first one, but reduced by rabifrequency*HBAR
    # factor = 1 seems to be optimal
    energy_pulse2 = detuning - factor * HBAR*rf_max
    # print("energy1: {:.4f}meV, energy2: {:.4f}meV".format(detuning, energy_pulse2))

    f,_,states = tls_commons.twopulse(t0=-t0, dt=dt, t_end=t0-dt,area1=area1, area2=area2, tau1=tau1, tau2=tau2, chirp1=0, chirp2=0, energy1=detuning, energy2=energy_pulse2, t02=t02)
    return f, states, t

# f, s, t = test_twopulse(dt=5, tau1=3500, tau2=3000, area1=20*np.pi, area2=10*np.pi)
# plt.plot(t,s)
# plt.ylim([-0.1,1.1])
# plt.show()
# f, s, t = test_twopulse(dt=5, tau1=3500, tau2=2495,t02=-252.5, area1=20*np.pi, area2=10*np.pi)
# plt.plot(t,s)
# plt.ylim([-0.1,1.1])
# plt.show()

# f, s, t = test_twopulse(dt=5, tau1=3700, tau2=5700,t02=700, area1=22.5*np.pi, area2=22*np.pi)
# f,s,t = test_twopulse(dt=1, tau1=5600, tau2=7600, area1=29.7613*np.pi, area2=24.3561*np.pi, t02=0)
# plt.plot(t,s)
# plt.ylim([-0.1,1.1])
# plt.show()

f,s,t = test_twopulse(dt=1, tau1=6192, tau2=9583, area1=29.0*np.pi, area2=29.0*np.pi, t02=-1812)
plt.plot(t,s)
plt.ylim([-0.1,1.1])
plt.show()

# n = 50
# x_ax = np.linspace(-1000,1000,n)  # t02
# y_ax = np.linspace(2000,4000,n)  # tau2
# z_ax = np.linspace(2000,4000,n)  # tau1
# ar1 = np.linspace(0, 30*np.pi, n)
# ar2 = np.linspace(0, 30*np.pi, n)

# factors = np.linspace(-1.5,1.5,1000)
# e = np.empty([len(factors)])
# for i in range(len(factors)):
#     e[i],_,_ = test_twopulse(factor=factors[i], dt=5, tau1=3500, tau2=2495,t02=-252.5, area1=20*np.pi, area2=10*np.pi)
# ind = np.unravel_index(np.argmax(e, axis=None), e.shape)
# print("{}, f:{}".format(ind,factors[ind]))
# plt.plot(factors,e)
# plt.xlabel("a: E_2 = d - a * rabifreq * h")
# plt.ylabel("f")
# plt.show()


#endvals = np.empty([len(y_ax), len(x_ax)])
#for i in tqdm.trange(len(y_ax)):
#    for j in range(len(x_ax)):
#        endvals[i,j],_,_ = test_twopulse(t02=x_ax[j], dt=5, tau1=3500, tau2=y_ax[i], area1=20*np.pi, area2=10*np.pi)

# ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
# print("{}, t02:{:.4f}, tau2:{:.4f}".format(ind,x_ax[ind[0]],y_ax[ind[1]]))
# plt.clf()
# plt.xlabel("t0")
# plt.ylabel("tau2")
# plt.pcolormesh(x_ax, y_ax, endvals, shading='auto')
# plt.colorbar()
# plt.show()
