from detuned_excitation.frequency_modulation.fm import fm_pulsed_excitation, fm_pulsed_fortran
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from scipy.optimize import curve_fit

def fm_area(det1=-30, det2=12, plot=True):
    x = np.linspace(0,20,100)*np.pi
    y = np.empty_like(x)
    for i in range(len(x)):
        _,v,_ = fm_pulsed_fortran(area=x[i], detuning=det1, small_detuning=det2)
        y[i] = v[-1,0]
    popt, pcov = curve_fit(func, x, y, bounds=(8,15))
    if plot:
        plt.plot(x/np.pi,y)
        #plt.plot(x/np.pi,np.sin(x/10)**2)
        #plt.plot(x/np.pi,func(x,*popt))
        print(popt)
        plt.xlabel("Pulse area/pi")
        plt.ylabel("final occupation")
        plt.show()
    return popt[0]

def func(x,omega):
    return np.sin(x/omega)**2

fm_area()

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

#omega_of_detune()

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
    plt.xlabel("detuning1 (meV)")
    plt.ylabel("detuning2 (meV)")
    plt.pcolormesh(x_ax, y_ax, endvals, shading='auto')
    plt.plot(x_ax[ind[1]],y_ax[ind[0]], 'r.')
    plt.colorbar()
    plt.show()
    return x_ax, y_ax, endvals

det1 = np.linspace(-60,-5,50)
det2 = np.linspace(0,20,50)
# fm_detuning(det1, det2)
# fm_pulsed_excitation(tau=10000, dt=1, area=7*np.pi, detuning=-6.58, small_detuning=18.22,plot=True)
# fm_pulsed_excitation(tau=10000, dt=1, area=7*np.pi, detuning=-59.44, small_detuning=17.17,plot=True)

def fm_area(areas, taus, detuning=-20, small_detuning=5, dt=1):
    """
    test the stability with regards to the two detunings
    """
    x_ax = taus
    y_ax = areas
    endvals = np.empty([len(y_ax), len(x_ax)])
    for i in tqdm.trange(len(y_ax)):
       for j in range(len(x_ax)):
           _,s,_ = fm_pulsed_fortran(tau=x_ax[j], dt=dt, area=y_ax[i], detuning=detuning, small_detuning=small_detuning)
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

areas = np.linspace(0,20*np.pi,50)
taus = np.linspace(1000,14000,50)
#fm_area(areas, taus)

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
    plt.xlabel("det1 (mev)")
    plt.ylabel("area/pi")
    plt.pcolormesh(x_ax, y_ax/np.pi, endvals, shading='auto')
    plt.plot(x_ax[ind[1]],y_ax[ind[0]]/np.pi, 'r.')
    plt.colorbar()
    plt.show()
    return x_ax, y_ax, endvals

fm_area_det(areas,det2)