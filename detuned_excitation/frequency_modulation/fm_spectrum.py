from detuned_excitation.frequency_modulation.fm import fm_pulsed_excitation, fm_pulsed_fortran
import numpy as np
from numpy.fft.helper import fftfreq
from detuned_excitation.two_level_system import pulse, helper
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import special
import tqdm

HBAR = 6.582119514e2  # meV fs


def fm_pulsed_spectrum(tau=5000, dt=1, area=6*np.pi, detuning=-12, small_detuning=4, factor=1.0, modulation_energy=None):
    t0 = -8 * tau
    t1 = 8 * tau
    s = int((t1 - t0) / dt)
    t = np.linspace(t0, t1, s + 1)
    p = pulse.Pulse(tau=tau, e_start=0, w_gain=0, e0=area)
    detuning_f = detuning/HBAR
    small_det_f = small_detuning/HBAR
    rf = lambda t: np.sqrt(factor*(p.get_envelope_f()(t))**2 + detuning_f**2)
    rf_0 = rf(0)
    if modulation_energy is not None:
        rf_0 = modulation_energy/HBAR
    # dont really need the freq. here, just the phase
    # freq = lambda t: detuning_f + small_det_f*np.sin(rf_0*t)
    # the phase is the integral of the frqeuency
    phase = lambda t : (detuning_f * t + (1/rf_0) * small_det_f * np.cos(rf_0*t))
    p.set_phase(phase)
    pulse_total = np.array([p.get_total(v) for v in t])
    pulse_total = pulse_total
    # now fft the pulse
    f = np.fft.fft(pulse_total)
    f = np.fft.fftshift(f)
    fft_freqs = 2*np.pi * HBAR * np.fft.fftfreq(len(pulse_total))
    fft_freqs = np.fft.fftshift(fft_freqs)
    df = np.abs(fft_freqs[0]-fft_freqs[1])
    integral = np.sum(np.abs(f)*df)
    integral_only_zero = 0
    # reduced_freq = []
    # reduced_f = []
    for i in range(len(fft_freqs)):
        if np.abs(fft_freqs[i]) < 1:  # below 3 meV
            integral_only_zero += np.abs(f[i])*df
            # reduced_freq.append[fft_freqs[i]]
            # reduced_f.append[f[i]]
    print("integral: {:.4f}, only zero peak: {:.4f}, area * ratio: {:.4f} pi".format(integral, integral_only_zero, (area/np.pi)*integral_only_zero/integral))
    modulation_index = np.abs(small_det_f / rf_0)
    print(modulation_index)
    print("j0(mu):{:.4f}, j1(mu):{:.4f}".format(special.j0(modulation_index),special.j1(modulation_index)))
    print("j1(mu)*area/pi : {:.4f}".format(special.j1(modulation_index)*area/np.pi))
    plt.plot(fft_freqs, np.abs(f))
    plt.xlabel("detuning in meV")
    plt.show()

# fm_pulsed_spectrum(tau=5000, dt=1, area=30*np.pi, detuning=-12, small_detuning=4)
# fm_pulsed_spectrum(tau=3500, dt=1, area=6*np.pi, detuning=-12, small_detuning=4)
# fm_pulsed_spectrum(tau=9000, dt=1, area=4*np.pi, detuning=-3, small_detuning=1.5)
# fm_pulsed_spectrum(tau=5000, area=6*np.pi, detuning=-12, small_detuning=4)
# fm_pulsed_spectrum(tau=12000, area=7*np.pi, detuning=-8, small_detuning=5)
# fm_pulsed_spectrum(tau=3000, area=33*np.pi, detuning=-12, small_detuning=4, factor=0.23)
fm_pulsed_spectrum(tau=3000, area=15*np.pi, detuning=-12, small_detuning=4, factor=1.0)
fm_pulsed_spectrum(tau=3000, area=30*np.pi, detuning=-6, small_detuning=1, modulation_energy=9.65)

def fm_rectangle_spectrum(tau=3500, dt=1, area=6*np.pi, detuning=-12, small_detuning=4):
    t0 = -8 * tau
    t1 = 8 * tau
    s = int((t1 - t0) / dt)
    t = np.linspace(t0, t1, s + 1)
    p = pulse.RectanglePulse(tau=tau, e_start=0, w_gain=0, e0=area)
    detuning_f = detuning/HBAR
    small_det_f = small_detuning/HBAR
    rf = lambda t: np.sqrt((0*p.get_envelope_f()(t))**2 + detuning_f**2)
    rf_0 = rf(0)
    # dont really need the freq. here, just the phase
    # freq = lambda t: detuning_f + small_det_f*np.sin(rf_0*t)
    # the phase is the integral of the frqeuency
    phase = lambda t : (detuning_f * t + (1/rf_0) * small_det_f * np.cos(rf_0*t))
    p.set_phase(phase)
    pulse_total = np.array([p.get_total(v) for v in t])
    pulse_total = pulse_total
    # now fft the pulse
    f = np.fft.fft(pulse_total)
    f = np.fft.fftshift(f)
    fft_freqs = 2*np.pi * HBAR * np.fft.fftfreq(len(pulse_total))
    fft_freqs = np.fft.fftshift(fft_freqs)
    df = np.abs(fft_freqs[0]-fft_freqs[1])
    integral = np.sum(np.abs(f)*df)
    integral_only_zero = 0
    # reduced_freq = []
    # reduced_f = []
    for i in range(len(fft_freqs)):
        if np.abs(fft_freqs[i]) < 1:  # below 3 meV
            integral_only_zero += np.abs(f[i])*df
            # reduced_freq.append[fft_freqs[i]]
            # reduced_f.append[f[i]]
    print("integral: {:.4f}, only zero peak: {:.4f}, area * ratio: {:.4f} pi".format(integral, integral_only_zero, (area/np.pi)*integral_only_zero/integral))
    
    modulation_index = np.abs(small_detuning / rf_0)
    print("j0(mu):{:.4f}, j1(mu):{:.4f}".format(special.j0(modulation_index),special.j1(modulation_index)))
    
    plt.plot(fft_freqs, np.abs(f))
    plt.show()

fm_rectangle_spectrum(tau=3500, dt=1, area=6*np.pi, detuning=-12, small_detuning=4)

def custom_colormap(plot=False):
    cdict = {
          'red':   ((0.0, 0.0, 0.0),
                    (0.125, 0.0, 0.0),
                    (0.25, 0.96, 0.96),
                    (0.375, 0.0, 0.0),
                    (0.50, 0.96, 0.96),
                    (0.625, 0.0, 0.0),
                    (0.75, 0.5, 0.5),
                    (0.875, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),

          'green': ((0.0, 0.0, 0.0),
                    (0.125, 0.0, 0.0),
                    (0.25, 0.89, 0.89),
                    (0.375, 0.0, 0.0),
                    (0.50, 0.3, 0.3),
                    (0.625, 0.0, 0.0),
                    (0.75, 0.89, 0.89),
                    (0.875, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),

          'blue':  ((0.0, 0.0, 0.0),
                    (0.125, 0.0, 0.0),
                    (0.25, 0.27, 0.27),
                    (0.375, 0.0, 0.0),
                    (0.50, 0.27, 0.27),
                    (0.625, 0.0, 0.0),
                    (0.75, 0.27, 0.27),
                    (0.875, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),
          }
    mycm = LinearSegmentedColormap("MyCM", cdict)
    if plot:
        x = np.arange(0, np.pi, 0.1)
        y = np.arange(0, 2 * np.pi, 0.1)
        X, Y = np.meshgrid(x, y)
        Z = np.cos(X) * np.sin(Y) * 10
        plt.pcolormesh(x,y,Z,shading='auto',cmap=mycm)
        plt.colorbar()
        plt.show()
    return mycm

# custom_colormap(plot=True)


def effective_area(tau, area, n=200, detuning1=-12, detuning2=4, save=False):
    y_ax = np.linspace(0,30,n)  # detuning 1
    x_ax = np.linspace(0,30,n)  # detuning 2
    areas = np.empty([n,n])
    arguments = np.empty_like(areas)
    rf_0 = area/(np.sqrt(2*np.pi)*tau)
    # we can calculate the 'effective pulse area' by using the amplitude of
    # the first sideband which is located at omega_carrier +- omega_signal
    # and has an amplitude which can be calculated by using the bessel function of the 
    # first kind, where the argument is the modulation index det2/sqrt(det1^2+rf_0^2)
    # eff.area = ara * j1(det2/sqrt(det1^2+rf_0^2)) 
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           new_freq = np.sqrt((y_ax[i]/HBAR)**2 + rf_0**2)
           areas[i,j] = special.j1((x_ax[j]/HBAR) / new_freq) * area / np.pi
           arguments[i,j] = (x_ax[j]/HBAR) / new_freq
    
    plt.title("argument of the bessel function")
    plt.pcolormesh(x_ax,y_ax,arguments,shading='auto')
    plt.xlabel("detuning 2 (meV)")
    plt.ylabel("-detuning 1 (meV)")
    plt.colorbar()
    plt.show()

    plt.title("effective areas")
    # plt.pcolormesh(x_ax,y_ax,1-np.abs(1-areas),shading='auto', vmax=1)
    plt.pcolormesh(x_ax,y_ax,areas,shading='auto')
    det1 = detuning1/HBAR
    det2 = detuning2/HBAR
    new_freq = np.sqrt(det1**2 + rf_0**2)
    print(special.j1(det2 / new_freq) * area / np.pi)
    plt.xlabel("detuning 2 (meV)")
    plt.ylabel("-detuning 1 (meV)")
    plt.colorbar()
    plt.show()

    # clip from 0 to 4
    areas2 = np.clip(areas,0,4)
    mycm = custom_colormap()
    plt.title("effective pulse area/pi")
    plt.pcolormesh(x_ax,y_ax,areas,shading='auto',cmap=mycm, vmin=0, vmax=4)
    plt.xlabel("detuning 2 (meV)")
    plt.ylabel("-detuning 1 (meV)")
    plt.colorbar()
    plt.show()

    # 
    analytic_final_occ = np.sin(areas*np.pi/2)**2
    plt.title("analytic final occupation")
    plt.pcolormesh(x_ax,y_ax,analytic_final_occ,shading='auto',vmin=0, vmax=1)
    plt.xlabel("detuning 2 (meV)")
    plt.ylabel("-detuning 1 (meV)")
    plt.colorbar()
    plt.show()

    if save:
        helper.save_colormap("analytic_occupation.csv", x_ax,y_ax,analytic_final_occ)

    # # we do not even need to to the following commented code,
    # # the corresponding numeric undetuned endvalue is exactly the same as the analytic above.
    # # we can compare it to our scheme though, as done below this comment-block
    # endvals = np.empty([n,n])
    # for i in tqdm.trange(len(y_ax)):
    #    for j in range(len(x_ax)):
    #        # compare to resonant excitation with effective pulsearea
    #        _,x,_ = fm_pulsed_fortran(tau,area=np.abs(areas[i,j])*np.pi,detuning=0,small_detuning=0)
    #        endvals[i,j] = x[-1,0].real
    # plt.pcolormesh(x_ax,y_ax,endvals,shading='auto')
    # plt.title("corresponding undetuned excitation endvalue")
    # plt.xlabel("detuning 2 (meV)")
    # plt.ylabel("-detuning 1 (meV)")
    # plt.colorbar()
    # plt.show()
    #
    # # now compute the difference of the analtic and the numeric result
    # diff = analytic_final_occ - endvals
    # plt.pcolormesh(x_ax,y_ax,diff,shading='auto', cmap=plt.get_cmap("coolwarm"), vmin=-0.2, vmax=0.2)
    # plt.title("diff. in values: analytic - numeric")
    # plt.xlabel("detuning 2 (meV)")
    # plt.ylabel("-detuning 1 (meV)")
    # plt.colorbar()
    # plt.show()
    endvals = pulsed_fm_endvalue(tau, area, n)
    if save:
        helper.save_colormap("numeric_occupation.csv", x_ax,y_ax,endvals)

    # now compute the difference of the analtic and the numeric result
    diff = analytic_final_occ - endvals
    plt.pcolormesh(x_ax,y_ax,diff,shading='auto', cmap=plt.get_cmap("coolwarm"), vmin=-1, vmax=1)
    plt.title("diff. in values: analytic - numeric_fm_scheme")
    plt.xlabel("detuning 2 (meV)")
    plt.ylabel("-detuning 1 (meV)")
    plt.colorbar()
    plt.show()

def pulsed_fm_endvalue(tau,area,n=100):
    y_ax = np.linspace(0,30,n)  # detuning 1
    x_ax = np.linspace(0,30,n)  # detuning 2
    endvals = np.empty([n,n])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           _,x,_ = fm_pulsed_fortran(tau,area=area,detuning=-y_ax[i],small_detuning=x_ax[j])
           endvals[i,j] = x[-1,0].real
    plt.pcolormesh(x_ax,y_ax,endvals,shading='auto')
    plt.title("final occupation FM-scheme")
    plt.xlabel("detuning 2 (meV)")
    plt.ylabel("-detuning 1 (meV)")
    plt.colorbar()
    plt.show()
    return endvals

def pulsed_fm_endvalue_dets(det1,det2,n=100):
    y_ax = np.linspace(0,20*np.pi,n)  # area
    x_ax = np.linspace(1000,20000,n)  # tau
    endvals = np.empty([n,n])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           _,x,_ = fm_pulsed_fortran(tau=x_ax[j],area=y_ax[i],detuning=det1,small_detuning=det2)
           endvals[i,j] = x[-1,0].real
    plt.pcolormesh(x_ax,y_ax,endvals,shading='auto')
    plt.title("final occupation FM-scheme")
    plt.xlabel("tau (fs)")
    plt.ylabel("area/pi")
    plt.colorbar()
    plt.show()
    return endvals

# effective_area(5000,6*np.pi,200)
# pulsed_fm_endvalue(5000,6*np.pi,100)

def effective_area_tau_area(det1, det2, n=200, save=False):
    y_ax = np.linspace(0,80*np.pi,n)  # area
    x_ax = np.linspace(1000,20000,n)  # tau
    areas = np.empty([n,n])
    arguments = np.empty_like(areas)
    
    # we can calculate the 'effective pulse area' by using the amplitude of
    # the first sideband which is located at omega_carrier +- omega_signal
    # and has an amplitude which can be calculated by using the bessel function of the 
    # first kind, where the argument is the modulation index det2/sqrt(det1^2+rf_0^2)
    # eff.area = ara * j1(det2/sqrt(det1^2+rf_0^2)) 
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           rf_0 = y_ax[i]/(np.sqrt(2*np.pi)*x_ax[j])
           new_freq = np.sqrt((det1/HBAR)**2 + rf_0**2)
           areas[i,j] = special.j1((det2/HBAR) / new_freq) * y_ax[i] / np.pi
           arguments[i,j] = (det2/HBAR) / new_freq
    
    plt.title("argument of the bessel function")
    plt.pcolormesh(x_ax,y_ax/np.pi,arguments,shading='auto')
    plt.xlabel("tau (fs)")
    plt.ylabel("area/pi")
    plt.colorbar()
    plt.show()

    plt.title("effective areas")
    # plt.pcolormesh(x_ax,y_ax,1-np.abs(1-areas),shading='auto', vmax=1)
    plt.pcolormesh(x_ax,y_ax/np.pi,areas,shading='auto')
    plt.xlabel("tau (fs)")
    plt.ylabel("area/pi")
    plt.colorbar()
    plt.show()

    # clip from 0 to 4
    areas2 = np.clip(areas,0,4)
    mycm = custom_colormap()
    plt.title("effective pulse area/pi")
    plt.pcolormesh(x_ax,y_ax/np.pi,areas,shading='auto',cmap=mycm, vmin=0, vmax=4)
    plt.xlabel("tau (fs)")
    plt.ylabel("area/pi")
    plt.colorbar()
    plt.show()

    # 
    analytic_final_occ = np.sin(areas*np.pi/2)**2
    plt.title("analytic final occupation")
    plt.pcolormesh(x_ax,y_ax/np.pi,analytic_final_occ,shading='auto',vmin=0, vmax=1)
    plt.xlabel("tau (fs)")
    plt.ylabel("area/pi")
    plt.colorbar()
    plt.show()

    if save:
        helper.save_colormap("analytic_occupation.csv", x_ax,y_ax/np.pi,analytic_final_occ)

    # endvals = pulsed_fm_endvalue_dets(det1, det2, n)
    # if save:
    #    helper.save_colormap("numeric_occupation.csv", x_ax,y_ax/np.pi,endvals)

    # # now compute the difference of the analtic and the numeric result
    # diff = analytic_final_occ - endvals
    # plt.pcolormesh(x_ax,y_ax/np.pi,diff,shading='auto', cmap=plt.get_cmap("coolwarm"), vmin=-1, vmax=1)
    # plt.title("diff. in values: analytic - numeric_fm_scheme")
    # plt.xlabel("tau (fs)")
    # plt.ylabel("area/pi")
    # plt.colorbar()
    # plt.show()

# effective_area_tau_area(-12,4,200)
#pulsed_fm_endvalue_dets(-12,4,100)