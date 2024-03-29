import numpy as np
import matplotlib.pyplot as plt
from detuned_excitation.frequency_modulation.fm import fm_pulsed_fortran, fm_rect_pulse


HBAR = 6.582119514e2  # meV fs

def h_twols(t, pulse):
    """
    two level system hamiltonian, rotating frame with light frequency
    since this is what we use to compute the fm-stuff, we also have to use it here
    """
    # since we just set the energy of the excited state to zero, the pulse
    # energy is the detuning
    detuning = HBAR * pulse.get_frequency(t)
    h = np.array([[0,0],
                  [0,-detuning]])

    omm = pulse.get_envelope(t)
    h += 0.5 * HBAR * np.array([[0,omm],[omm,0]])
    return h



def twolevels_dressedstates(tau=10000, dt=4, area=7*np.pi, detuning=-10, small_detuning=3, rect=False):
    t,x,pulse = fm_pulsed_fortran(tau, dt, area, detuning, small_detuning)
    if rect:
        t,x,pulse = fm_rect_pulse(tau, dt, area, detuning, small_detuning)
    states = np.empty([len(t),2])
    states[:,0] = 1-x[:,0].real
    states[:,1] = x[:,0].real
    polars = x[:,1]

    plt.plot(t*1e-3,states[:,0],label="g")
    plt.plot(t*1e-3,states[:,1],label="x")
    plt.legend()
    plt.show()
    e_values = np.empty([len(t),2])
    e_vectors = np.empty([len(t),2,2], dtype=complex)
    for i in range(len(t)):
        e_values[i],e_vectors[i] = np.linalg.eigh(h_twols(t[i],pulse))
    
    # first fix the phase of the eigenvectors
    for i in range(len(t)):
        # if first component of first EV is not real and smaller than 0:
        # multiply all EVs with exp(-1j*angle)
        angle=0
        if (np.imag(e_vectors[i,0,0] !=0 or e_vectors[i,0,0] < 0)):
            angle = np.angle(e_vectors[i,0,0])
        e_vectors[i,:,:] = e_vectors[i,:,:]*np.exp(-1j*angle)
    
    for i in range(2):
        # this prints for all the eigenvectors the composition of the states
        print("{} start: ({:.2f},{:.2f}), end: ({:.2f},{:.2f})".format(i, *np.real(e_vectors[0,:,i]),*np.real(e_vectors[-1,:,i])))

    
    for i in range(2):
        plt.plot(t*1e-3,e_values[:,i],label="E_{}".format(i))
    plt.legend()
    plt.xlabel("t in ps")
    plt.ylabel("E in meV")
    plt.show()

    n_colors = np.empty([6,e_values.shape[0]])  # for gnuplot
    s_colors = []
    r_array = np.array([0,255])/255
    g_array = np.array([0,0])/255
    b_array = np.array([255,0])/255
    a_array = np.array([255,255])/255
    a_array_gp = np.array([0,0])  # for gnuplot

    for i in range(2):
        colors = []
        for j in range(e_values.shape[0]):
            e = np.abs(e_vectors[j,:,i])**2
            r = int(np.clip(np.dot(r_array,e),0,1)*255)
            g = int(np.clip(np.dot(g_array,e),0,1)*255)
            b = int(np.clip(np.dot(b_array,e),0,1)*255)
            a = int(np.clip(np.dot(a_array,e),0,1)*255)
            agp = int(np.clip(np.dot(a_array_gp,e),0,1)*255)
            n_colors[i,j] = 65536*r + 256*g + b + agp*16777216
            colors.append("#{:02x}{:02x}{:02x}{:02x}".format(r,g,b,a))
        plt.scatter(t,e_values[:,i],c=colors)
        s_colors.append(colors)
    plt.show()

    dressed_states = np.empty([len(t),2])
    for j in range(len(t)):
        for i in range(2):
            dressed_states[j,i] = np.dot(np.abs(e_vectors[j,:,i])**2, states[j])
            dressed_states[j,i] += 2*np.real(e_vectors[j,0,i]*np.conj(e_vectors[j,1,i]*polars[j]))
    
    for i in range(2):
        plt.plot(t*1e-3,dressed_states[:,i],label="state_{}".format(i))
    plt.legend()
    plt.show()


    
# twolevels_dressedstates(tau=9000, area=4*np.pi, detuning=-3, small_detuning=1.5)
# twolevels_dressedstates(tau=2000, area=4*np.pi, detuning=0.1, small_detuning=0)
twolevels_dressedstates(3000, dt=1, area=5*np.pi, detuning=-12, small_detuning=4, rect=True)
