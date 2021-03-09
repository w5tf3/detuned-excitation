from detuned_excitation.frequency_modulation.fm import fm_rect_pulse, fm_pulsed_excitation
import matplotlib.pyplot as plt
import numpy as np
import tqdm

#t, x1, p1 = fm_pulsed_excitation(tau=9000, area=4*np.pi, detuning=-3, small_detuning=1.5)
#plt.plot(t,x1[:,0].real)
#plt.show()

t, x, p = fm_rect_pulse(3000, dt=1, area=5*np.pi, detuning=-12, small_detuning=4)
#plt.plot(t, x[:,0].real)
#plt.show()
