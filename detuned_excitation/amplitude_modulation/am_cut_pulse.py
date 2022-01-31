import numpy as np
import matplotlib.pyplot as plt
from numpy import fft
from numpy.fft.helper import fftfreq
from detuned_excitation.amplitude_modulation.am import get_detuning2
from detuned_excitation.two_level_system.helper import nm_to_mev, mev_to_nm
from detuned_excitation.two_level_system import tls_commons, pulse
import tqdm

HBAR = 6.582119514e2  # meV fs

def cut_super(dt=5, tau1=2400, tau2=3040, area1=22.65*np.pi, area2=19.29*np.pi, t02=0, detuning1=-8.0000, detuning2=-19.1630):
    """
    for checking how exactly the filtering works:
    We put in the known to work SUPER-Parameters and FT the pulses, then use this as a mask in the frequency domain
    and transform it back.
    """
    t0 = 8*tau2
    t = np.arange(-t0,t0+dt,dt)
    n_steps = len(t)
    e_length = 2*n_steps - 1
    t_new = np.linspace(-t0,t0,e_length)
    dt_new = np.abs(t_new[0]-t_new[1])
    p1 = pulse.Pulse(tau=tau1, e_start=detuning1, w_gain=0, e0=area1)
    p2 = pulse.Pulse(tau=tau2, e_start=detuning2, w_gain=0, e0=area2)
    e01 = p1.get_total(t_new)
    e02 = p2.get_total(t_new)
    pulse_total = e01+e02
    f = np.fft.fft(pulse_total)
    # f = np.fft.fftshift(f)
    fft_freqs = 2*np.pi*HBAR*np.fft.fftfreq(len(pulse_total),d=dt_new)
    fft_freqs = np.fft.fftshift(fft_freqs)
    plt.plot(fft_freqs, np.abs(np.fft.fftshift(f)))
    plt.xlim(0,30)
    plt.show()
    n_steps = len(t)
    pulse_t = np.fft.ifft(f)
    plt.plot(np.abs(pulse_t))
    plt.show()
    f,p,states,polars = tls_commons.tls_arbitrary_pulse(t[0], pulse_t, n_steps, dt=dt, strict=False)
    t = np.linspace(t[0],t[-1],len(states))
    plt.plot(t, states)
    plt.show()
    return t, states, f, pulse_t

# _,_,_,pulse_t = cut_super()
# detuning1=-8
# max_r = np.max(np.abs(pulse_t))
# print("max_r: {:.5f}, max_rabi_energy: {:.5f}".format(max_r,max_r*HBAR))
# rf_max = np.sqrt(max_r**2 + (detuning1/HBAR)**2)
# print("detuning2: {:.5f}".format(detuning1 - HBAR*rf_max))

def cut_pulses(t0=-22000, t1=22000, area=120*np.pi, center=-7.5, fwhm=20, detuning1=-5, detuning2=-10, cut_width1=2, cut_width2=2, factor1=1.0, factor2=1.0, dt=4, do_plot=False, gauss=False, background=None):
    """
    ### Prameters:
    area: 'area' of the whole pulse
    center: central energy of the pulse in meV
    fwhm: fwhm of the spectrum in meV
    detuning1: center of the first slit of the modulator in meV
    detuning2: center of the second slit of the modulator in meV
    cut_width: slit width for cutting, in meV

    ### Comment:
    We do not have to worry too much about the sign of the detunings 
    as long as we do net consider i.e. phonons.
    The SUPER scheme works the same with positive detunings as with negative ones.
    """
    # for the temporal representation of the pulses
    # tau = 4000
    # t0 = -4 * tau
    # t1 = 4 * tau
    #s = int((t1 - t0) / dt)
    #t = np.linspace(t0, t1, s + 1)
    t = np.arange(t0,t1,dt)
    # spectrum in meV
    sigma = fwhm / (2*np.sqrt(2*np.log(2)))
    fft_freqs = 2*np.pi * HBAR * np.fft.fftfreq(len(t),d=dt)
    spectrum = area / (np.sqrt(2*np.pi)*sigma) * np.exp(-(fft_freqs-center)**2/(2*sigma**2))

    # define the correct mask to multiply the spectrum with
    # for now, we just use a rectangular slit.
    alpha = 100
    mask1 = 0
    mask2 = 0
    if not gauss:
        mask1 = 1/( (1 + np.exp(-alpha*(fft_freqs-(detuning1+cut_width1/2))) ) * (1 + np.exp(-alpha*(-fft_freqs + (detuning1-cut_width1/2)) )))
        mask2 = 1/( (1 + np.exp(-alpha*(fft_freqs-(detuning2+cut_width2/2))) ) * (1 + np.exp(-alpha*(-fft_freqs + (detuning2-cut_width2/2)) )))

    if gauss:
        mask1 = 1/np.sqrt(2*np.pi*cut_width1**2)*np.exp(-0.5*((fft_freqs-detuning1)/cut_width1)**2)  # 1/( (1 + np.exp(-alpha*(fft_freqs-(detuning1+cut_width1/2))) ) * (1 + np.exp(-alpha*(-fft_freqs + (detuning1-cut_width1/2)) )))
        mask2 = 1/np.sqrt(2*np.pi*cut_width2**2)*np.exp(-0.5*((fft_freqs-detuning2)/cut_width2)**2)  # 1/( (1 + np.exp(-alpha*(fft_freqs-(detuning2+cut_width2/2))) ) * (1 + np.exp(-alpha*(-fft_freqs + (detuning2-cut_width2/2)) )))
    
    mask1 = factor1 * mask1/np.max(mask1)
    mask2 = factor2 * mask2/np.max(mask2)
    mask = mask1 + mask2

    if background != None:
        for i in range(len(fft_freqs)):
            if fft_freqs[i] < detuning1 and fft_freqs[i] > detuning2:
                if mask[i] < background:
                    mask[i] = background

    # temporal representation
    pulse_t = np.fft.ifft(spectrum)
    pulses_t = np.fft.ifft(mask*spectrum)
    if do_plot:
        fig,axes = plt.subplots(3,2,constrained_layout=True)
        axes[0,0].plot(np.fft.fftshift(fft_freqs),np.fft.fftshift(spectrum),label="in spectrum")
        # axes[0].set_xlabel("time in ps")
        axes[0,0].set_ylabel("intensity")
        axes[0,0].set_xlim((-50,50))
        axes[0,0].legend()

        axes[1,0].plot(np.fft.fftshift(fft_freqs),np.fft.fftshift(mask),label="filter mask")
        axes[1,0].set_xlim((-50,50))
        axes[1,0].set_ylabel("transmission")
        axes[1,0].legend()

        axes[2,0].set_xlabel("Energy (meV)")
        axes[2,0].set_xlim((-50,50))
        axes[2,0].set_ylabel("intensity")
        axes[2,0].plot(np.fft.fftshift(fft_freqs),np.fft.fftshift(spectrum*mask),label="out spectrum")
        axes[2,0].legend()

        axes[0,1].plot(t/1000, np.abs(np.fft.ifftshift(pulse_t)), label='in pulse')
        axes[0,1].legend()
        axes[1,1].plot(t/1000, np.abs(np.fft.ifftshift(pulses_t)), label='out pulses')
        axes[1,1].legend()
        #axes[1,1].set_xlabel("time (ps)")

        n_steps = int((len(t)-2)/2)
        _,_,states,_ = tls_commons.tls_arbitrary_pulse(t[0], np.fft.ifftshift(pulses_t), n_steps, dt=2*dt, strict=False)
        t_ = np.linspace(t[0],t[-1],len(states))
        axes[2,1].plot(t_,states, label="time dynamics")
        axes[2,1].legend()
        axes[2,1].set_xlabel("time (ps)")

        # axes[3].set_xlim((-3000,3000))
        plt.show()
    return t, np.fft.ifftshift(pulses_t)


# t, pulse = cut_pulses(do_plot=True, center=-20,detuning1=-5, detuning2=-11, cut_width1=0.4,cut_width2=0.4, factor2=0.4,gauss=True)
# detuning1 = -5
# cut_width = 0.4
# t, pulse_t = cut_pulses(area=350*np.pi, do_plot=True, center=-20,detuning1=detuning1, detuning2=-11.40211, cut_width1=cut_width, cut_width2=cut_width, factor2=0., gauss=True)
# max_r = np.max(np.abs(pulse_t))
# print("max_r: {:.5f}, max_rabi_energy: {:.5f}".format(max_r,max_r*HBAR))
# rf_max = np.sqrt(max_r**2 + (detuning1/HBAR)**2)
# print("detuning2: {:.5f}".format(detuning1 - HBAR*rf_max))
# plt.plot(t, np.abs(pulse_t))
# plt.show()
cut_pulses(gauss=True, do_plot=True, area=300*np.pi, center=-20, detuning1=-5, detuning2=-13.6735, cut_width1=0.4, cut_width2=0.4, factor2=0.3796)
cut_pulses(background=0.05, gauss=True, do_plot=True, area=300*np.pi, center=-20, detuning1=-5, detuning2=-13.0204, cut_width1=0.4, cut_width2=0.4, factor2=0.5429)

# cut_pulses(do_plot=True, gauss=False, area=300*np.pi, center=-20, detuning1=-5, detuning2=-13.6735, cut_width1=1.2, cut_width2=1.2, factor2=0.2)


def use_cut_pulse(dt, area=120*np.pi, center=-20,detuning1=-5, detuning2=-11, cut_width1=0.8, cut_width2=0.8, factor2=0.4, gauss=True, background=None):
    #t0 = 5*tau
    #t = np.arange(-t0,t0+dt,dt)
    t, pulse = cut_pulses(area=area, dt=dt/2, do_plot=False, center=center,detuning1=detuning1, detuning2=detuning2, cut_width1=cut_width1, cut_width2=cut_width2, factor2=factor2, gauss=gauss, background=background)
    n_steps = int((len(t)-2)/2)
    # plt.plot(t,np.abs(pulse))
    # plt.show()
    #e_length = 2*n_steps - 1
    #t_new = np.linspace(-t0,t0,e_length)
    #_p = pulse.Pulse(tau=tau, e_start=energy, w_gain=0, e0=area)
    #e0 = _p.get_total(t_new)
    f,p,states,polars = tls_commons.tls_arbitrary_pulse(t[0], pulse, n_steps, dt=dt, strict=False)
    t = np.linspace(t[0],t[-1],len(states))
    # plt.plot(t, states)
    # plt.show()
    return t, states, f

# t, s, _ = use_cut_pulse(dt=8, area=450*np.pi, center=-20, detuning1=-5, detuning2=-12.7755, cut_width1=0.4, cut_width2=0.4, factor2=0.6082)
# plt.plot(t,s)
# plt.show()
# t, s, _ = use_cut_pulse(dt=8, area=300*np.pi, center=-20, detuning1=-5, detuning2=-13.6735, cut_width1=0.4, cut_width2=0.4, factor2=0.3796)
# plt.plot(t,s)
# plt.show()
# t, s, _ = use_cut_pulse(gauss=False, dt=8, area=300*np.pi, center=-20, detuning1=-5, detuning2=-13.6735, cut_width1=1.2, cut_width2=1.2, factor2=0.2)
# plt.plot(t,s)
# plt.show()

def det2_factor2(n=50, dt=8, area=900*np.pi, center=20, detuning1=-5, cut_width1=0.8, cut_width2=0.8, gauss=True, background=None):
    detuning2 = np.linspace(-10,-14,n)
    factor2 = np.linspace(0.2,1,n)  #factor2=np.linspace(0,1,50)
    x_ax = detuning2
    y_ax = factor2
    endvals = np.zeros([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
            _,_,endvals[i,j] = use_cut_pulse(background=background, gauss=gauss, dt=dt, area=area, center=center, detuning1=detuning1, detuning2=x_ax[j], cut_width1=cut_width1, cut_width2=cut_width2, factor2=y_ax[i])
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    max_x,max_y = x_ax[ind[1]],y_ax[ind[0]]
    print("{}, detuning2:{:.4f}, factor2:{:.4f}, endval:{:.4f}".format(ind,max_x,max_y,endvals[ind[0],ind[1]]))
    plt.xlabel("detuning2")
    plt.ylabel("transmission_2")
    plt.pcolormesh(x_ax, y_ax, endvals, shading='auto')
    plt.plot(x_ax[ind[1]],y_ax[ind[0]], 'r.')
    plt.colorbar()
    plt.show()
    return x_ax, y_ax, endvals

x,y,z = det2_factor2(background=0.2,area=300*np.pi, center=-20, detuning1=-5, cut_width1=0.4, cut_width2=0.4)
plt.xlabel("transmission_2")
plt.ylabel("exciton occupation")
plt.plot(y, z[:,48])
plt.show()
# det2_factor2(gauss=False, area=300*np.pi, center=-20, detuning1=-5, cut_width1=1.2, cut_width2=1.2)


def det2_area(dt=8, center=-20, detuning1=-5, cut_width1=0.8, cut_width2=0.8, factor2=0.5):
    detuning2 = np.linspace(2*detuning1+0.1,2*detuning1-2,100)
    area = np.linspace(20,200,100)*np.pi  #factor2=np.linspace(0,1,50)
    x_ax = detuning2
    y_ax = area  #factor2
    endvals = np.zeros([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
            _,_,endvals[i,j] = use_cut_pulse(dt=dt, area=y_ax[i], center=center, detuning1=detuning1, detuning2=x_ax[j], cut_width1=cut_width1, cut_width2=cut_width2, factor2=factor2)
            
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    max_x,max_y = x_ax[ind[1]],y_ax[ind[0]]
    print("{}, detuning2:{:.4f}, area:{:.4f}, endval:{:.4f}".format(ind,max_x,max_y,endvals[ind[0],ind[1]]))
    plt.xlabel("detuning2")
    plt.ylabel("area")
    plt.pcolormesh(x_ax, y_ax, endvals, shading='auto')
    plt.plot(x_ax[ind[1]],y_ax[ind[0]], 'r.')
    plt.colorbar()
    plt.show()
    return x_ax, y_ax, endvals

# det2_area(cut_width1=1.2,cut_width2=1.2)
