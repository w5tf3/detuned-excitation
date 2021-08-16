from detuned_excitation.frequency_modulation.fm import * #fm_pulsed_excitation, fm_pulsed_fortran, fm_rect_pulse
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.optimize import curve_fit

HBAR = 6.582119514e2  # meV fs

def fm_area(tau, det1=-30, det2=12, plot=True):
    x = np.linspace(0,20,100)*np.pi
    y = np.empty_like(x)
    for i in range(len(x)):
        _,v,_ = fm_pulsed_fortran(tau=tau, area=x[i], detuning=det1, small_detuning=det2)
        y[i] = v[-1,0]
    if plot:
        plt.plot(x/np.pi,y)
        #plt.plot(x/np.pi,np.sin(x/10)**2)
        #plt.plot(x/np.pi,func(x,*popt))
        plt.xlabel("Pulse area/pi")
        plt.ylabel("final occupation")
        plt.show()

def fm_rect_area(areas, dt=4, tau=10000, detuning=-10, small_detuning=3, rect_modul=False):
    x = areas
    y = np.empty_like(areas)
    for i in tqdm.trange(len(x)):
        _, v, _ = fm_rect_pulse(tau=10000, dt=dt, area=x[i], detuning=detuning, small_detuning=small_detuning, phase=0, plot=False, rect_modul=rect_modul)
        y[i] = v[-1,0].real
    ind = np.unravel_index(np.argmax(y, axis=None), y.shape)
    print("{}, area:{:.4f} pi, occupation:{:.4f}".format(ind,x[ind[0]]/np.pi,y[ind[0]]))
    plt.xlabel("occupation")
    plt.ylabel("area/pi")
    plt.plot(x/np.pi,y)
    plt.plot(x[ind[0]]/np.pi,y[ind[0]], 'r.')
    plt.show()

def func(x,omega):
    return np.sin(x/omega)**2

def omega_of_detune():
    det1 = np.linspace(-40,-10,50)
    det2 = np.linspace(4,15,50)
    x_ax = det1
    y_ax = det2
    endvals = np.empty([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           endvals[i,j] = fm_area(det1=x_ax[j], det2=y_ax[i],plot=False)
    
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    print("{}, det1:{:.4f} meV, det2:{:.4f} meV".format(ind,x_ax[ind[1]],y_ax[ind[0]]))
    plt.xlabel("detuning1 (meV)")
    plt.ylabel("detuning2 (meV)")
    plt.pcolormesh(x_ax, y_ax, endvals, shading='auto')
    plt.plot(x_ax[ind[1]],y_ax[ind[0]], 'r.')
    plt.colorbar()
    plt.show()

def fm_detuning(det1, det2, tau=10000, area=7*np.pi, dt=1):
    """
    test the stability with regards to the two detunings
    """
    x_ax = det1
    y_ax = det2
    endvals = np.empty([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           _,s,_ = fm_pulsed_fortran(tau=tau, dt=dt, area=area, detuning=x_ax[j], small_detuning=y_ax[i])
           endvals[i,j] = s[-1,0].real
    
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    print("{}, det1:{:.4f} meV, det2:{:.4f} meV".format(ind,x_ax[ind[1]],y_ax[ind[0]]))
    print("final occupation: {:.4f}".format(endvals[ind[0],ind[1]]))
    plt.xlabel("detuning1 (meV)")
    plt.ylabel("detuning2 (meV)")
    plt.pcolormesh(x_ax, y_ax, endvals, shading='auto')
    plt.plot(x_ax[ind[1]],y_ax[ind[0]], 'r.')
    plt.colorbar()
    plt.show()
    return x_ax, y_ax, endvals

def fm_areas(areas, taus, detuning=-20, small_detuning=5, dt=1, rect=False):
    """
    test the stability with regards to the two detunings
    """
    x_ax = taus
    y_ax = areas
    s = 0
    endvals = np.empty([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
            if not rect:
                _,s,_ = fm_pulsed_fortran(tau=x_ax[j], dt=dt, area=y_ax[i], detuning=detuning, small_detuning=small_detuning)
            if rect:
                _,s,_ = fm_rect_pulse(tau=x_ax[j], dt=dt, area=y_ax[i], detuning=detuning, small_detuning=small_detuning, plot=False)
            endvals[i,j] = s[-1,0].real
    
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    print("{}, tau:{:.4f} fs, area:{:.4f} pi".format(ind,x_ax[ind[1]],y_ax[ind[0]]/np.pi))
    plt.xlabel("tau (fs)")
    plt.ylabel("area/pi")
    plt.pcolormesh(x_ax, y_ax/np.pi, endvals, shading='auto')
    plt.plot(x_ax[ind[1]],y_ax[ind[0]]/np.pi, 'r.')
    plt.colorbar()
    plt.show()
    return x_ax, y_ax, endvals

def fm_area_det(areas, det2, tau=10000,detuning=-40,dt=1):
    """
    test the stability with regards to the two detunings
    """
    x_ax = det2
    y_ax = areas
    endvals = np.empty([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           _,s,_ = fm_pulsed_fortran(tau=tau, dt=dt, area=y_ax[i], detuning=detuning, small_detuning=x_ax[j])
           endvals[i,j] = s[-1,0].real
    
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    print("{}, det2:{:.4f} meV, area:{:.4f} pi".format(ind,x_ax[ind[1]],y_ax[ind[0]]/np.pi))
    print("final occupation: {:.4f}".format(endvals[ind[0],ind[1]]))
    plt.xlabel("det1 (mev)")
    plt.ylabel("area/pi")
    plt.pcolormesh(x_ax, y_ax/np.pi, endvals, shading='auto')
    plt.plot(x_ax[ind[1]],y_ax[ind[0]]/np.pi, 'r.')
    plt.colorbar()
    plt.show()
    return x_ax, y_ax, endvals

def fm_factor(tau, area, detuning, small_detuning, dt=1, negative=0.0,n=20, max_factor=2):
    factors = np.linspace(negative,max_factor,n)
    endvals = np.empty_like(factors)
    rf_0 = area/(np.sqrt(2*np.pi)*tau)
    det1 = detuning/HBAR
    det2 = small_detuning/HBAR
    rabi_freqs = np.sqrt(det1**2 + factors*rf_0**2)
    for i in tqdm.trange(len(factors)):
        _,x,_ = fm_pulsed_fortran(tau=tau, dt=dt, area=area, detuning=detuning, small_detuning=small_detuning, factor=factors[i])
        endvals[i] = x[-1,0].real
    rabi_energies = HBAR * rabi_freqs
    ind = np.unravel_index(np.argmax(endvals), endvals.shape)
    print("endvalue:{:.4f} at factor:{:.4f}".format(endvals[ind[0]], factors[ind[0]]))
    plt.plot(factors,endvals)
    plt.xlabel("factors")
    plt.ylabel("final occupation")
    plt.show()
    plt.plot(rabi_energies,endvals)
    plt.xlabel("rabi energies (meV)")
    plt.ylabel("final occupation")
    plt.show()

def fm_energy(tau, area, detuning, small_detuning, dt=1,n=20, factor=0.3):
    energies = np.linspace(detuning - factor*np.abs(detuning),detuning + factor*np.abs(detuning),n)
    endvals = np.empty_like(energies)
    for i in tqdm.trange(len(energies)):
        _,x,_ = fm_pulsed_fortran(tau=tau, dt=dt, area=area, detuning=detuning, small_detuning=small_detuning, modulation_energy=energies[i])
        endvals[i] = x[-1,0].real
    ind = np.unravel_index(np.argmax(endvals), endvals.shape)
    print("endvalue:{:.4f} at energy:{:.4f}".format(endvals[ind[0]], energies[ind[0]]))
    plt.plot(energies,endvals)
    plt.xlabel("modulation_energy (meV)")
    plt.ylabel("final occupation")
    plt.show()
    plt.plot(energies-detuning,endvals)
    plt.xlabel("modulation_energy - detuning1 (meV)")
    plt.ylabel("final occupation")
    plt.show()

def fm_energy_area(tau, area, detuning, small_detuning, dt=1,n=20,n2=20, factor=0.3):
    energies = np.linspace(np.abs(detuning)*0.8,np.abs(detuning) + factor*np.abs(detuning),n)
    areas = np.linspace(0,area,n2)
    x_ax = energies
    y_ax = areas
    endvals = np.empty([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           _,x,_ = fm_pulsed_fortran(tau=tau, dt=dt, area=y_ax[i], detuning=detuning, small_detuning=small_detuning, modulation_energy=x_ax[j])
           endvals[i,j] = x[-1,0].real
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    print("{}, (meV:) modulation_energy={:.4f}, area={:.4f}*np.pi".format(ind,x_ax[ind[1]],y_ax[ind[0]]/np.pi))
    print("final occupation: {:.4f}".format(endvals[ind[0],ind[1]]))
    plt.xlabel("modulation energy (mev)")
    plt.ylabel("area/pi")
    plt.ylim(0,area/np.pi)
    plt.pcolormesh(x_ax, y_ax/np.pi, endvals, shading='auto')
    plt.plot(x_ax[ind[1]],y_ax[ind[0]]/np.pi, 'r.')
    plt.plot(x_ax, (1/np.pi)*np.sqrt(2*np.pi*tau**2)*(1/HBAR)*np.sqrt(-detuning**2 + x_ax**2), 'r-')
    plt.colorbar()
    plt.show()
    plt.xlabel("detuning sideband (mev)")
    plt.ylabel("area/pi")
    plt.ylim(0,area/np.pi)
    plt.xlim(x_ax[0]+detuning,x_ax[-1]+detuning)
    plt.pcolormesh(x_ax+detuning, y_ax/np.pi, endvals, shading='auto')
    plt.plot(x_ax[ind[1]]+detuning,y_ax[ind[0]]/np.pi, 'r.')
    plt.plot(x_ax, (1/np.pi)*np.sqrt(2*np.pi*tau**2)*(1/HBAR)*np.sqrt(-detuning**2 + (x_ax-detuning)**2), 'r-')
    plt.colorbar()
    plt.show()
    return x_ax, y_ax/np.pi, endvals

def fm_search_optimum(tau, dt, area, detuning, small_detuning, n=10, percent=2):
    low = (100-percent)/100
    high = (100+percent)/100
    det1 = detuning
    det2 = small_detuning
    taus = np.linspace(tau*low,tau*high,n)
    areas = np.linspace(area*low,area*high,n)
    det1s = np.linspace(det1*low,det1*high,n)
    det2s = np.linspace(det2*low,det2*high,n)
    endvals = np.empty([n,n,n,n])
    for v1 in tqdm.trange(n):
        for v2 in range(n):
            for v3 in range(n):
                for v4 in range(n):
                    _,x,_ = fm_pulsed_fortran(dt=dt,tau=taus[v1],area=areas[v2],detuning=det1s[v3],small_detuning=det2s[v4])
                    endvals[v1,v2,v3,v4] = x[-1,0].real
    ind = np.unravel_index(np.argmax(endvals, axis=None), endvals.shape)
    v1,v2,v3,v4 = ind[0],ind[1],ind[2],ind[3]
    m_tau = taus[v1]
    m_area = areas[v2]
    m_det1 = det1s[v3]
    m_det2 = det2s[v4]
    print("endavlue: {:.4f}".format(endvals[v1,v2,v3,v4]))
    print("tau={:.3f},area={:.3f}*np.pi,detuning={:.3f},small_detuning={:.3f}".format(m_tau,m_area/np.pi,m_det1,m_det2))
