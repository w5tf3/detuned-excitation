from os import stat
import numpy as np
from detuned_excitation.two_level_system import tls
from detuned_excitation.two_level_system import sixls
# print(tls.__doc__)


HBAR = 6.582119514e2  # meV fs

class Electric_field_envelope():
    
    def __init__(self, area, tau):
        self.area = area
        self.tau = tau

    def e_f(self, t): 
        """
        Pulse: gaussian with sqrt of variance tau and area 'area' for times t as a function
        """
        return (self.area / (np.sqrt(2 * np.pi) * self.tau)) * np.exp(-0.5 * (t / self.tau)**2)

    def e_f_dot(self, t):
        return (self.area / (np.sqrt(2 * np.pi) * self.tau)) * (-t/self.tau**2)*np.exp(-0.5 * (t / self.tau)**2)
    
    def e_f_normed(self, t):
        """
        returns e_f normed to 0...1
        """
        return np.exp(-0.5 * (t/self.tau)**2)

def two_level(t0, dt, t_end, f_start, p_start, energy, area, pulse_tau):
    """
    returns f,p for end timestep where f is real and p is complex.
    uses fortran to solve time propagation of a two level system.
    Constant laser energy, uses rotating frame with system frequency.
    This means it works with small timesteps especially for small detunings.
    """
    n_steps = int(abs(t_end-t0)/dt)+1
    # print(n_steps)
    # print(t0+dt*(n_steps-1))
    # print(t)
    # tls solves up until t_0+dt*(n_steps-1) so that in total n_steps values are returned
    # including the starting value 
    # since n_steps = int(abs(t_end-t0)/dt)+1, it solves up to t_end, if (t_end-t_0)%dt=0 
    delta_e = 0 #1000  # meV
    f, p, _ = tls.tls(t0, dt, n_steps, f_start, p_start, energy+delta_e, area, delta_e, pulse_tau)
    return f, p

def two_level_chirp_a(t0=-4*400, dt=20, t_end=4*400, f_start=0, p_start=0, energy=0, a_chirp=60e-6, area=8*np.pi, pulse_tau=400):
    """
    give a chirp a (1/fs^2), start and end endtime in fs 
    pulse used is gaussian with variance pulse_tau. start/end should be at least +- 4*pulse_tau
    energy is an additional detuning, i.e. this is the pulse energy at t=0.

    """
    n_steps = int(abs(t_end-t0)/dt)+1
    # print(n_steps)
    # print(t0+dt*(n_steps-1))
    # print(t)
    # tls solves up until t_0+dt*(n_steps-1) so that in total n_steps values are returned
    # including the starting value 
    delta_e = 0 # 1000  # meV
    f, p, states = tls.tls_chirp_a(t0, dt, n_steps, f_start, p_start, pulse_tau, energy+delta_e, a_chirp, area, delta_e)
    return f, p, states


def twopulse(t0=-4*400, dt=4, t_end=4*400, f_start=0, p_start=0, t02=0, tau1=400, tau2=400, energy1=0, energy2=0, chirp1=60e-6, chirp2=0, area1=7*np.pi, area2=0):
    """
    use the fortran implementation compiled with f2py to solve the diff. eqs. for two simultaneous pulses
    the default parameters show a chirped excitation with one pulse
    """
    n_steps = int(abs(t_end-t0)/dt)+1
    f, _, states, polars = tls.tls_twopulse(t_0=t0, dt=dt, n_steps=n_steps, in_state=f_start,
                                in_polar=p_start, tau1=tau1, tau2=tau2, e_energy1=energy1,
                                e_energy2=energy2, a_chirp1=chirp1, a_chirp2=chirp2,
                                e01=area1, e02=area2, delta_e=0, t02=t02)
    return f, polars, states


def twopulse_cw(t0=-4*400, dt=4, t_end=4*400, f_start=0, p_start=0, t02=0, tau1=400, energy1=0, energy2=0, chirp1=60e-6, area1=7*np.pi, area2=0):
    """
    use the fortran implementation compiled with f2py to solve the diff. eqs. for two simultaneous pulses
    the default parameters show a chirped excitation with one pulse
    the second pulse is a cw with amplitude area2, so it is not normalized to the simulation length.
    """
    n_steps = int(abs(t_end-t0)/dt)+1
    f, _, states, polars = tls.tls_twopulse_cw_second(t_0=t0, dt=dt, n_steps=n_steps, in_state=f_start,
                                in_polar=p_start, tau1=tau1, e_energy1=energy1,
                                e_energy2=energy2, a_chirp1=chirp1, a_chirp2=0.0,
                                e01=area1, e02=area2, delta_e=0, t02=t02)
    mean_f = np.mean(states[-int(len(states)/10):])
    return mean_f, polars, states


def two_level_fm(tau=10000,dt=4,detuning=-10,detuning_small=3,area=7*np.pi,fm_freq=0.015217):
    t0 = -4*tau
    t1 = 4*tau
    n_steps = int((t1 - t0) / dt) + 1
    in_state=0
    in_polar=0+0j
    f,p,states,polars = tls.tls_fm(t0,dt,n_steps,in_state,in_polar,tau,detuning,detuning_small,area,fm_freq)
    return f, p, states, polars


def six_levels_mixed_polar(t_0, t_end, dt=10, tau1=3500, tau2=3500, energy_1=1.0, energy_2=1.0, e01=1*np.pi, e02=0*np.pi, t02=0.0, polar_m1=1.0, polar_m2=1.0, bx=0, bz=0, state_param=np.array([1, 0, 0, 0, 0, 0]), polarizations_param=np.zeros([15], dtype=complex), delta_B=-0.25, d0=0.25, d1=0.12, d2=0.05, delta_E=0.0):
    """
    
    """
    t = np.arange(t_0, t_end, dt)
    n_steps = int(abs(t_end - t_0)/dt)

    # light_polarization = 0 is purely negative circular polarized light, l_p = 1 purely positive cpl
    # polar_p**2 + polar_m**2 = 1
    energy_1 += delta_E
    energy_2 += delta_E
    endstate,endpolar,states,polars = sixls.sixls_twopulse(t_0, dt, n_steps, state_param, polarizations_param, polar_m1, polar_m2, tau1, tau2, energy_1, energy_2, e01, e02, bx, bz, delta_B, d0, d1, d2, delta_E, t02)
    return t, states, endstate, endpolar, polars


def runge_kutta(t0, x0, t1, h, equation, pulse, delta_e):
    """
    runge kutta 4 to solve differential equations.
    :param t0: time at which the calculation starts
    :param x0: initial values
    :param t1: time to stop
    :param h: step length
    :param equation: equation to solve
    :param pulse: electric field
    :param delta_e: 2-niveau energy difference
    :return:
    """
    s = int((t1 - t0) / h)
    n_ = len(x0)
    t = np.linspace(t0, t1, s + 1)
    x = np.zeros([s + 1, n_], dtype=complex)
    x[0] = x0
    for t_, i in zip(t, range(s)):
        k1 = equation(t_, x[i], pulse, delta_e)
        k2 = equation(t_ + h / 2, x[i] + k1 * h / 2, pulse, delta_e)
        k3 = equation(t_ + h / 2, x[i] + k2 * h / 2, pulse, delta_e)
        k4 = equation(t_ + h / 2, x[i] + k3 * h, pulse, delta_e)
        x[i + 1] = x[i] + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6.
    return t, np.array(x, dtype=complex)


def bloch_eq(t, x, pulse, delta_e):
    # eq. in frame rotating with laser frequency
    e_f = pulse.get_envelope(t)
    delta = pulse.get_frequency(t) - delta_e/HBAR
    f = x[0]
    p = x[1]

    _f = np.imag(np.conj(e_f)*p)
    _p = 1j*delta*p + 0.5j*e_f*(1-2*f)
    return np.array([_f, _p], dtype=complex)


def bloch_eq_constrf(t, x, pulse_, rf_freq=0):
    # eq. in frame rotating with a constant frequency
    # if rf_freq=0, it is rotating with the system frequency
    # the np.exp(1j*delta_e/HBAR*t) factor results from the RF
    e_f = pulse_.get_total(t)
    delta = rf_freq
    f = x[0]
    p = x[1]

    _f = np.imag( np.conj(e_f) * p )
    _p = 1j * delta * p + 0.5j * e_f * ( 1 - 2 * f ) 
    return np.array( [_f, _p], dtype=complex )


def bloch_eq_general_rf(t, x, pulse_, _):
    # eq. in frame rotating with a constant frequency
    # if rf_freq=0, it is rotating with the system frequency
    # the np.exp(1j*delta_e/HBAR*t) factor results from the RF
    e_f = pulse_.get_total(t)
    delta = pulse_.rf_freq(t)
    f = x[0]
    p = x[1]

    _f = np.imag( np.conj(e_f) * p )
    _p = 1j * delta * p + 0.5j * e_f * ( 1 - 2 * f ) 
    return np.array( [_f, _p], dtype=complex )