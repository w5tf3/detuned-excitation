import twolevelsystem as tl
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def fm_area():
    x = np.linspace(0,15,40)
    y = np.empty_like(x)
    for i in tqdm.trange(len(x)):
        _,v,_ = tl.fm_pulsed_excitation(area=x[i]*np.pi)
        y[i] = v[-1,0]
    plt.plot(x,y)
    plt.xlabel("Pulse area/pi")
    plt.ylabel("final occupation")
    plt.show()

fm_area()