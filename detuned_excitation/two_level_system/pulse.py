import numpy as np
import matplotlib.pyplot as plt

HBAR = 6.582119514e-2  # meV fs


class Pulse:

    def __init__(self, tau, e_start, w_gain=0, t0=0, e0=1):
        self.tau = tau  # in fs
        self.e_start = e_start  # in meV
        self.w_start = e_start / HBAR  # in 1 / fs
        self.w_gain = float(w_gain)  #  in 1/fs^2
        self.t0 = t0
        self.e0 = e0
        self.freq = None

    def __repr__(self):
        return "%s(tau=%r, e_start=%r, w_gain=%r, t0=%r, e0=%r)" % (
            self.__class__.__name__, self.tau, self.e_start, self.w_gain, self.t0, self.e0
        )

    def get_w0(self):
        return self.w_start

    def get_wg_t0(self):
        return 0.5*self.w_gain * self.t0

    #def get_envelope(self):
    #    return lambda x: self.e0 * np.exp(-0.5 * ((x - self.t0) / self.tau) ** 2) / (np.sqrt(2 * np.pi) * self.tau)
    

    def get_envelope(self, t):
        return self.e0 * np.exp(-0.5 * ((t - self.t0) / self.tau) ** 2) / (np.sqrt(2 * np.pi) * self.tau)

    def get_envelope_f(self):
        return lambda t: self.e0 * np.exp(-0.5 * ((t - self.t0) / self.tau) ** 2) / (np.sqrt(2 * np.pi) * self.tau)

    def set_frequency(self, f):
        """
        use a lambda function f taking a time t to set the time dependent frequency.
        """
        self.freq = f

    def get_frequency(self, t):
        """
        phidot, i.e. the derivation of the phase,
        is the current frequency
        :return: frequency omega for a given time 
        """
        if self.freq is not None:
            return self.freq(t)
        return self.w_start + self.w_gain * (t - self.t0)


    def get_phase(self, t):
        """
        Carrier phase phi(t)/t :
        exp(i phi(t))
        phi(t) = (w_start+0.5*w_gain*t)*t
        :param t: time for which the phase/t should be returned
        :return: Carrier phase(t)/t, time dependent
        """
        return self.w_start + 0.5*self.w_gain * (t - self.t0)

    def get_full_phase(self,t):
        return self.w_start * (t - self.t0) + 0.5*self.w_gain * ((t - self.t0) **2)
    
    def get_energies(self):
        """
        get energy diff of +- tau for chirped pulse
        E=hbar*w
        if tau and everything is in fs, output in meV
        """
        low = self.get_frequency(-self.tau)
        high = self.get_frequency(self.tau)
        energy_range = np.abs(high-low)*HBAR  # meV
        return energy_range

    def get_total(self, t):
        return self.get_envelope(t) * np.exp(-1j * self.get_full_phase(t))

    def plot(self, t0, t1, n):
        t = np.linspace(t0, t1, n)
        y = self.get_total(t)
        y2 = self.get_envelope(t)
        # y3 = np.cos(self.get_carrier()(t) * (t - self.t0))
        plt.plot(t, y.real, 'r-')
        plt.plot(t, y.imag, 'b-')
        plt.plot(t, y2, 'g-')
        # plt.plot(t,y3,'y.')
        plt.show()


class RectanglePulse(Pulse):
    """
    Pulse with rectangular pulse shape, with amplitude area/tau from -tau/2 to tau/2, else 0.
    frequency is constant or chirped (though this is not really needed).
    but can be set using set_frequency and a lambda function.
    """

    def get_envelope_f(self):
        return lambda t: self.e0/self.tau if np.abs(t-self.t0) < self.tau/2 else 0

    def get_envelope(self, t):
        return self.get_envelope_f()(t)

    def plot(self, t0, t1, n):
        t = np.linspace(t0, t1, n)
        y = [self.get_total(i) for i in t]
        y2 = [self.get_envelope(i) for i in t]
        # y3 = np.cos(self.get_carrier()(t) * (t - self.t0))
        plt.plot(t, y, 'b.')
        plt.plot(t, y2, 'r.')
        # plt.plot(t,y3,'y.')
        plt.show()


class ChirpedPulse(Pulse):
    def __init__(self, tau_0, e_start, alpha=0, t0=0, e0=1*np.pi):
        self.tau_0 = tau_0
        self.alpha = alpha
        super().__init__(np.sqrt(alpha**2 / tau_0**2 + tau_0**2), e_start, w_gain=alpha/(alpha**2 + tau_0**4), t0=t0, e0=e0)
    
    def get_parameters(self):
        """
        returns tau and chirp parameter
        """
        return "tau: {:.4f} fs , a: {:.4f} ps^-2".format(self.tau, self.w_gain*(1000**2))

    def get_envelope(self, t):
        return self.e0 * np.exp(-0.5 * ((t - self.t0) / self.tau) ** 2) / (np.sqrt(2 * np.pi * self.tau * self.tau_0))

    def get_ratio(self):
        """
        returns ratio of pulse area chirped/unchirped: tau / sqrt(tau * tau_0)
        """
        return np.sqrt(self.tau / self.tau_0)


class MultiPulse:
    def __init__(self, pulse1, pulse2):
        self.pulse1 = pulse1
        self.pulse2 = pulse2

    def get_total(self, t):
        """
        returns e0_1(t)*exp(-i*phi1(t)) + e0_2(t)*exp(-i*phi2(t))
        """
        _p1 = self.pulse1.get_total(t)  # get_envelope(t) * np.exp(-1j*self.pulse1.get_full_phase(t))
        _p2 = self.pulse2.get_total(t)  # get_envelope(t) * np.exp(-1j*self.pulse2.get_full_phase(t))
        return _p1 + _p2