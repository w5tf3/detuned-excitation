import numpy as np
import matplotlib.pyplot as plt

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



beat()