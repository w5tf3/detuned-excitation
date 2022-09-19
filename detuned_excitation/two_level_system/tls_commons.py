from os import stat
import numpy as np
from detuned_excitation.two_level_system.tls import f90 as tls
from detuned_excitation.two_level_system.sixls import f90 as sixls
from detuned_excitation.two_level_system.biexciton import f90 as biexciton
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


def twopulse(t0=-4*400, dt=4, t_end=4*400, f_start=0, p_start=0, t02=0, tau1=400, tau2=400, energy1=0, energy2=0, chirp1=60e-6, chirp2=0, area1=7*np.pi, area2=0, phase=0, rf_energy=None):
    """
    use the fortran implementation compiled with f2py to solve the diff. eqs. for two simultaneous pulses
    the default parameters show a chirped excitation with one pulse
    """
    n_steps = int(abs(t_end-t0)/dt)+1
    if rf_energy is None:
        rf_energy = energy1
    f, _, states, polars = tls.tls_twopulse(t_0=t0, dt=dt, n_steps=n_steps, in_state=f_start,
                                in_polar=p_start, tau1=tau1, tau2=tau2, e_energy1=energy1,
                                e_energy2=energy2, a_chirp1=chirp1, a_chirp2=chirp2,
                                e01=area1, e02=area2, rf_energy=rf_energy, t02=t02, phase=phase)
    return f, polars, states


def twopulse_cw(t0=-4*400, dt=4, t_end=4*400, f_start=0, p_start=0, t02=0, tau1=400, energy1=0, energy2=0, chirp1=60e-6, area1=7*np.pi, area2=0, slope=2e3):
    """
    use the fortran implementation compiled with f2py to solve the diff. eqs. for two simultaneous pulses
    the default parameters show a chirped excitation with one pulse
    the second pulse is a cw with amplitude area2, so it is not normalized to the simulation length.
    """
    n_steps = int(abs(t_end-t0)/dt)+1
    f, _, states, polars = tls.tls_twopulse_cw_second(t_0=t0, dt=dt, n_steps=n_steps, in_state=f_start,
                                in_polar=p_start, tau1=tau1, e_energy1=energy1,
                                e_energy2=energy2, a_chirp1=chirp1, a_chirp2=0.0,
                                e01=area1, e02=area2, delta_e=0, t02=t02, slope=slope)
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

def tls_fm_rectangular(tau, dt, det1, det2, amplitude, omega_mod, in_state=0, in_polar=0j):
    t_0 = -tau
    t_end = tau
    t = np.arange(t_0, t_end, dt)
    n_steps = int(abs(t_end - t_0)/dt)
    states, polars = tls.tls_fm_rectangular(t_0,tau,dt,n_steps,amplitude,det1,det2,omega_mod,in_state,in_polar)
    return t, states, polars

def six_levels_two_color(t_0, t_end, dt=10, tau1=3500, tau2=3500, energy_1=1.0, energy_2=1.0, e01=1*np.pi, e02=0*np.pi, t02=0.0, polar_m1=1.0, polar_m2=1.0, bx=0, bz=0, state_param=np.array([1, 0, 0, 0, 0, 0]), polarizations_param=np.zeros([15], dtype=complex), delta_b=-0.25, d0=0.25, d1=0.12, d2=0.05, delta_e=0.0):
    """
    
    """
    t = np.arange(t_0, t_end, dt)
    n_steps = int(abs(t_end - t_0)/dt)

    # light_polarization = 1.0 is purely negative circular polarized light, l_p = 0.0 purely positive cpl
    # polar_p**2 + polar_m**2 = 1
    energy_1 += delta_e
    energy_2 += delta_e
    endstate,endpolar,states,polars = sixls.sixls_twopulse(t_0, dt, n_steps, state_param, polarizations_param, polar_m1, polar_m2, tau1, tau2, energy_1, energy_2, e01, e02, bx, bz, delta_b, d0, d1, d2, delta_e, t02)
    return t, states, endstate, endpolar, polars


def biex_am_fortran(t0, t_end, tau1=10000, tau2=1000, alpha=0, dt=1, det1=0, det2=0, area1=10*np.pi, area2=0*np.pi,t02=0, delta_b=8, delta_e=0, phase=0, in_state=np.array([1,0,0]), in_polar=np.array([0,0,0],dtype=complex)):
    #tau = tau1 if tau1 > (tau2+np.abs(t02)) else (tau2+np.abs(t02))
    #t0 = -4*tau
    #t1 = 4*tau
    n_steps = int(abs(t_end-t0)/dt)+1
    f,p,states,polars = biexciton.biex_twopulse(t0,dt,n_steps,in_state,in_polar,tau1,tau2,det1,det2,area1,area2,delta_b,delta_e,t02,phase,alpha)
    return f, p, states, polars


def biex_rect(tau, det1, det2, area1, area2, dt=5, delta_b=8, delta_e=0, in_state=np.array([1,0,0]), in_polar=np.array([0,0,0],dtype=complex)):
    t0 = -0.6*tau
    t1 = 0.6*tau
    n_steps = int((t1 - t0) / dt) + 1
    f,p,states,polars = biexciton.biex_rectangle(t0,dt,n_steps,in_state,in_polar,tau,det1,det2,area1,area2,delta_b,delta_e)
    t = np.linspace(t0,t1,len(states))
    return t, f, p, states, polars

def tls_arbitrary_pulse(t0, e0, n_steps, dt=3, delta_e=0, in_state=0, in_polar=0j, strict=True):
    """
    ### Parameters 
    e0: array with electric field, complex. has to be of length 2*len(out_size)-1, where out_size is the (desired) size of the result array
    n_steps: just for checking the input
    dt: time step
    """
    if strict and 2*n_steps-1 != len(e0):
        print("size of e0 does not match step count")
        print("is:{}, should:{}".format(len(e0),2*n_steps-1))
        exit(1)
    e0 = e0[:2*n_steps-1]
    f,p,states,polars = tls.tls_arbitrary_field(t0,dt,in_state,in_polar,e0,delta_e,n_steps)
    return f, p, states, polars

def biex_arbitrary_pulse(e0, n_steps, dt=3, delta_e=0, delta_b=4, in_state=np.array([1,0,0]), in_polar=np.array([0,0,0], dtype=complex), strict=True):
    """
    ### Parameters 
    e0: array with electric field, complex. has to be of length 2*len(out_size)-1, where out_size is the (desired) size of the result array
    n_steps: just for checking the input
    dt: time step
    """
    if strict and 2*n_steps-1 != len(e0):
        print("size of e0 does not match step count")
        exit(1)
    f,p,states,polars = biexciton.biex_arbitrary_field(dt,in_state,in_polar,e0,delta_b,delta_e,n_steps)
    return f, p, states, polars

def runge_kutta(t0, x0, t1, h, equation, pulse, delta_e, args={}):
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
        k1 = equation(t_, x[i], pulse, delta_e, **args)
        k2 = equation(t_ + h / 2, x[i] + k1 * h / 2, pulse, delta_e, **args)
        k3 = equation(t_ + h / 2, x[i] + k2 * h / 2, pulse, delta_e, **args)
        k4 = equation(t_ + h, x[i] + k3 * h, pulse, delta_e, **args)
        x[i + 1] = x[i] + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6.
    return t, np.array(x, dtype=complex)

def euler(t0, x0, t1, h, equation, pulse, delta_e):
    s = int((t1 - t0) / h)
    n_ = len(x0)
    t = np.linspace(t0, t1, s + 1)
    x = np.zeros([s + 1, n_], dtype=complex)
    x[0] = x0
    for i in range(s):
        x[i + 1] = x[i] + h * equation(i, x[i], pulse[i], delta_e)
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
    # eq. in frame rotating with the laser frequency
    e_f = pulse_.get_total(t)
    delta = pulse_.rf_freq(t)
    f = x[0]
    p = x[1]

    _f = np.imag( np.conj(e_f) * p )
    _p = 1j * delta * p + 0.5j * e_f * ( 1 - 2 * f ) 
    return np.array( [_f, _p], dtype=complex )


def bloch_eq_pulse_total(t, x, pulse_, rf_freq=0):
    # eq. in frame rotating with a constant frequency
    # if rf_freq=0, it is rotating with the system frequency
    # the np.exp(1j*delta_e/HBAR*t) factor results from the RF
    e_f = pulse_
    delta = rf_freq
    f = x[0]
    p = x[1]

    _f = np.imag( np.conj(e_f) * p )
    _p = 1j * delta * p + 0.5j * e_f * ( 1 - 2 * f ) 
    return np.array( [_f, _p], dtype=complex )

def biexciton_constrf(t, _x, pulse_, rf_freq=0,delta_b=4):
    e_f = pulse_.get_total(t)
    delta = rf_freq

    E_X = 0
    E_B = 2*E_X - delta_b

    phidot = rf_freq * HBAR
    g = 1-_x[0]-_x[1]
    x = _x[0]
    b = _x[1]
    gx = _x[2]
    gb = _x[3]
    xb = _x[4]

    #g = (1/2)*1j*gx*np.conj(e_f) - 1/2*1j*e_f*np.conj(gx)
    _gx = (1/2)*1j*g*e_f + (1/2)*1j*gb*np.conj(e_f) + 1j*gx*(-E_X + phidot)/HBAR - 1/2*1j*e_f*x
    _gb = 1j*gb*(-E_B + 2*phidot)/HBAR + (1/2)*1j*gx*e_f - 1/2*1j*e_f*xb
    __x = -1/2*1j*gx*np.conj(e_f) + (1/2)*1j*e_f*np.conj(gx) - 1/2*1j*e_f*np.conj(xb) + (1/2)*1j*xb*np.conj(e_f)
    _xb = -1/2*1j*b*e_f - 1/2*1j*gb*np.conj(e_f) + (1/2)*1j*e_f*x + 1j*xb*(-E_B + E_X + phidot)/HBAR
    _b = (1/2)*1j*e_f*np.conj(xb) - 1/2*1j*xb*np.conj(e_f)
    return np.array( [__x, _b, _gx, _gb, _xb], dtype=complex )
