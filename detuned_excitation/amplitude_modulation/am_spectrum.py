import numpy as np
import matplotlib.pyplot as plt
from numpy import fft
from numpy.fft.helper import fftfreq
from detuned_excitation.amplitude_modulation.am import get_detuning2
from detuned_excitation.two_level_system.helper import nm_to_mev, mev_to_nm

HBAR = 6.582119514e2  # meV fs

def elec_field(t, amplitude, tau, t02, frequency, phase=0):
    """
    frequency as angular frequency omega
    """
    return amplitude / (np.sqrt(2*np.pi)*tau) * np.exp(-(t-t02)**2/(2*tau**2)) * np.exp(-1j*frequency*t + 1j*phase)


def am_spectrum(omega0=10, tau1=5000, tau2=5000, dt=5, area1=10*np.pi, area2=10*np.pi, detuning=-5, t02=0, phase=0, rectangular=False, factor=1.0, detuning2=None, ft_time=8):
    """
    rectangular and phase are not yet implemented
    """
    tau = tau1 if tau1 > (tau2+np.abs(t02)) else (tau2+np.abs(t02))
    t0 = -ft_time * tau
    t1 = ft_time * tau
    s = int((t1 - t0) / dt)
    t = np.linspace(t0, t1, s + 1)
    if detuning2 is None:
        detuning2 = get_detuning2(tau1, area1, detuning,rectangular,factor)
    print("detuning1:{:.4f}, detuning2:{:.4f}".format(detuning, detuning2))
    
    detuning_f = detuning/HBAR
    detuning2_f = detuning2/HBAR
    pulse1 = elec_field(t, area1, tau1, 0, omega0/HBAR + detuning_f)
    pulse2 = elec_field(t, area2, tau2, t02, omega0/HBAR + detuning2_f,phase=phase)
    pulse_total = pulse1 + pulse2
    pulse_total = pulse_total #np.abs(pulse_total)
    # now fft the pulse
    f = np.fft.fft(pulse_total)
    f = np.fft.fftshift(f)
    f1 = np.fft.fftshift(np.fft.fft(pulse1))
    f2 = np.fft.fftshift(np.fft.fft(pulse2))
    fft_freqs = 2*np.pi * HBAR * np.fft.fftfreq(len(pulse_total),d=dt)
    fft_freqs = np.fft.fftshift(fft_freqs)
    #df = np.abs(fft_freqs[0]-fft_freqs[1])
    #integral = np.sum(np.abs(f)*df)
    #integral_only_zero = 0
    # reduced_freq = []
    # reduced_f = []
    #for i in range(len(fft_freqs)):
    #    if np.abs(fft_freqs[i]) < 1:  # below 3 meV
    #        integral_only_zero += np.abs(f[i])*df
            # reduced_freq.append[fft_freqs[i]]
            # reduced_f.append[f[i]]
    #print("integral: {:.4f}, only zero peak: {:.4f}, area * ratio: {:.4f} pi".format(integral, integral_only_zero, (area/np.pi)*integral_only_zero/integral))
    plt.plot(-fft_freqs, np.abs(f)**2,label="total")
    plt.plot(-fft_freqs, np.abs(f1)**2,dashes=[6, 2],label="pulse1")
    plt.plot(-fft_freqs, np.abs(f2)**2,dashes=[6, 2],label="pulse2")
    plt.title("phase:{:.2f}pi".format(phase/np.pi))
    plt.ylabel("intensity (arb.)")
    plt.legend()
    plt.xlabel("detuning in meV")
    plt.xlim(-30,30)
    plt.show()
    return -fft_freqs, np.abs(f)

# am_spectrum(omega0=50, dt=1, tau1=6200, tau2=9600, area1=29.0*np.pi, area2=29.0*np.pi, detuning=-5.0, t02=-1800)
# am_spectrum(omega0=0, dt=1, tau1=100, tau2=100, area1=29.0*np.pi, area2=29.0*np.pi, detuning=-5.0, t02=-1800)
# am_spectrum(omega0=0, dt=3, tau1=2400, tau2=3040, area1=22.65*np.pi, area2=19.29*np.pi, t02=-730, detuning=-8.0)
am_spectrum(omega0=0, dt=3, tau1=1800, tau2=1800, area1=0.7*np.pi, area2=0.7*np.pi, t02=0, detuning=-0.25, detuning2=0.25,ft_time=16,phase=np.pi)
