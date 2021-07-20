from detuned_excitation.frequency_modulation.fm import fm_rect_pulse, fm_pulsed_excitation
from detuned_excitation.frequency_modulation.fm_stability import *
import matplotlib.pyplot as plt
import numpy as np
import tqdm

# t, x1, p1 = fm_pulsed_excitation(tau=9000, area=4*np.pi, detuning=-3, small_detuning=1.5, filename="intro_plot.csv")
# t, x1, p1 = fm_pulsed_excitation(tau=9000, area=4*np.pi, detuning=-3, small_detuning=0)
# t, x2, p1 = fm_pulsed_excitation(tau=9000, area=4*np.pi, detuning=-4, small_detuning=2, filename="intro_plot.csv")
# t, x2, p1 = fm_pulsed_excitation(tau=3000, area=6*np.pi, detuning=-12, small_detuning=4, factor=0.7368, modulation_energy=12.1129)
# t, x2, p1 = fm_pulsed_excitation(tau=3000, area=30*np.pi, detuning=-6, small_detuning=1, factor=0.7368, modulation_energy=9.65)
# t, x2, p1 = fm_pulsed_excitation(dt=1, tau=4000, area=30*np.pi, detuning=-6, small_detuning=1, modulation_energy=8.2575)
# t, x2, p1 = fm_cutoff(dt=4, tau=4000, area=30*np.pi, detuning=-6, small_detuning=1, modulation_energy=-8.2575, use_exp=True)
# t, x2, p1 = fm_cutoff(tau=4000, detuning=-6, small_detuning=2, modulation_energy=-6.1225, area=6.7347*np.pi, use_exp=True)
##t, x2, p1 = fm_cutoff(tau=4000, area=47.235*np.pi, detuning=-6, small_detuning=1, modulation_energy=+11.038)#, use_exp=True)
#t, x2, p1 = fm_pulsed_fortran(dt=1, tau=4000, area=30*np.pi, detuning=-6, small_detuning=1, modulation_energy=8.2575)

# t, x1, p1 = fm_pulsed_excitation(tau=12000, area=7*np.pi, detuning=-8, small_detuning=5)
# fm_area(9000,det1=-4, det2=2)
# t, x1, p1 = fm_pulsed_excitation(tau=5000, area=6*np.pi, detuning=-12, small_detuning=4)
# fm_factor(tau=3000, area=6*np.pi, detuning=-12, small_detuning=4)
# fm_energy(tau=4000,area=30*np.pi, detuning=-6, small_detuning=1, n=300,factor=1.5)


# fm_energy_area(tau=4000,area=60*np.pi, detuning=-6, small_detuning=0.5, dt=4, n=150,n2=150,factor=2)


# t, x2, p1 = fm_pulsed_excitation(tau=4000, detuning=-6, small_detuning=2, modulation_energy=8.1906, area=29.3960*np.pi)
# t, x2, p1 = fm_pulsed_excitation(tau=4000, area=49.57*np.pi, detuning=-6, small_detuning=1, modulation_energy=11.393)
t, x2, p1 = fm_pulsed_excitation(tau=4000, detuning=-6, small_detuning=2, modulation_energy=6.1225, area=6.7347*np.pi)
# t, x2, p1 = fm_pulsed_excitation(tau=4000, detuning=-6, small_detuning=1.5, modulation_energy=6.1758, area=8.7879*np.pi)
# t, x2, p1 = fm_pulsed_excitation(tau=4000, detuning=-6, small_detuning=1.5, modulation_energy=10.1879, area=52.7517*np.pi)
##t, x2, p1 = fm_pulsed_excitation(tau=4000, detuning=-6, small_detuning=2, modulation_energy=6.0843, area=6.2207*np.pi)
t, x2, p1 = fm_rect_pulse(rect_modul_rf=True,tau=4000, detuning=-8, small_detuning=2, area=6.2207*np.pi, rect_modul=True)
#t, x2, p1 = fm_pulsed_excitation(tau=4000, area=47.235*np.pi, detuning=-6, small_detuning=1, modulation_energy=11.038)
# fm_energy_area(tau=4000,area=60*np.pi, detuning=-12, small_detuning=2, n=50,n2=50,factor=1.0)
# t, x2, p1 = fm_pulsed_excitation(tau=4000, area=6.1224*np.pi, detuning=-6, small_detuning=2, modulation_energy=6.095)
#t, x1, p1 = fm_pulsed_excitation(tau=3500, area=5.9*np.pi, detuning=-12, small_detuning=4)
#t, x1, p1 = fm_pulsed_excitation(tau=5000, area=4.9*np.pi, detuning=-12, small_detuning=5)
# plt.plot(t,x1[:,1].real, 'r-')
# plt.plot(t, x1[:,1].imag, 'b-')
# t, x1, p1 = fm_pulsed_excitation(tau=1500, area=4.5*np.pi, detuning=-12, small_detuning=4, factor=1)
# # fm_factor(tau=9000, area=4*np.pi, detuning=-3, small_detuning=1.5)
# plt.plot(t,x1[:,0].real)

t, x2, p1 = fm_pulsed_excitation(tau=4000, detuning=-6, small_detuning=1, modulation_energy=8.448, area=37.16*np.pi)

plt.plot(t,x2[:,0].real)
plt.plot(t, [0 for v  in t])
plt.show()

# fm_search_optimum(dt=4, tau=1905.750,area=4.147*np.pi,detuning=-10.227,small_detuning=5.082, percent=5, n=10)
# fm_search_optimum(dt=4, tau=3500, area=4.81*np.pi, detuning=-12, small_detuning=4)
# t, x1, p1 = fm_pulsed_excitation(3000,area=33*np.pi,detuning=-12,small_detuning=4,factor=0.227)
# plt.plot(t,x1[:,0].real)
# plt.show()
# fm_factor(tau=5000, area=6*np.pi, detuning=-12, small_detuning=4,n=200, dt=4, max_factor=10)

# areas = np.linspace(4.7*np.pi,4.9*np.pi,10)
# fm_rect_area(areas,tau=3500, detuning=-12, small_detuning=4,rect_modul=True)

# t, x, p = fm_rect_pulse(3500, dt=1, area=4.81*np.pi, detuning=-12, small_detuning=4, filename="data.csv", rect_modul=True)
#fm_rect_pulse(3500, dt=1, area=6*np.pi, detuning=-12, small_detuning=4)

# areas = np.linspace(0,6*np.pi, 100)
# # # det1s = np.linspace(-3,-12,50)
# # det2s = np.linspace(1,1.5,50)
# taus = np.linspace(6000,6000, 1)
# fm_area(areas, taus, -3, 1.5)
# fm_area_det(areas, det2s, 9000, -3)
# fm_detuning(det1s, det2s, 6000,5*np.pi)

# t, x, p = fm_rect_pulse(3500, dt=1, area=6.0816*np.pi, detuning=-12, small_detuning=4, filename="data.csv", rect_modul=False)

# plt.plot(t, x[:,1].real, 'r-')
# plt.plot(t, x[:,1].imag, 'b-')
# plt.show()

# fact = np.linspace(0.9,1.1,20)
# e = []
# for f in fact:
#     _,x,_ = fm_rect_pulse(3000, dt=1, area=8*np.pi, detuning=-9, small_detuning=2.5, factor=f, plot=False)
#     e.append(x[-1,0].real)
# plt.plot(fact, e)
# plt.show()
