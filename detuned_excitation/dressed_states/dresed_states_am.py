import numpy as np
import matplotlib.pyplot as plt
from detuned_excitation.amplitude_modulation.am import am_twocolor, am_twocolor_fortran
from detuned_excitation.two_level_system import pulse as ps
from detuned_excitation.two_level_system.helper import export_csv
import tqdm

HBAR = 6.582119514e2  # meV fs

def h_twols(t, pulse_total, rf_freq):
    """
    two level system hamiltonian, rotating frame with light frequency
    since this is what we use to compute the fm-stuff, we also have to use it here
    """
    # since we just set the energy of the excited state to zero, the pulse
    # energy is the detuning
    # detuning = HBAR * rf_freq
    # h = np.array([[0,0],
    #               [0,-detuning]], dtype=complex)

    # omm = pulse_total.get_total(t)
    # h += 0.5 * HBAR * np.array([[0,np.conj(omm)],[omm,0]])

    delta_e = 0
    h = np.diag([0., delta_e])
    
    phidot = pulse_total.get_frequency(t)

    om = pulse_total.get_envelope(t)
    h -= HBAR * np.diag([0.,phidot])
    # in rotating frame, only the amplitudes matter
    h += 0.5 * HBAR *np.array([[0,-om],
                               [-om,0]])
    # detuning = HBAR * pulse_total.get_frequency(t)
    # h = np.array([[0,0],
    #               [0,-detuning]])

    # omm = pulse_total.get_envelope(t)
    # h += 0.5 * HBAR * np.array([[0,omm],[omm,0]])
    return h



def twolevels_dressedstates(tau1=6200, tau2=9600, dt=10, area1=29*np.pi, area2=29*np.pi, t02=-1800, detuning1=-5, detuning2=None, filename=None, t_factor=4):
    # important: use rotating frame with detuning (frequency) of pulse1
    rf_energy = detuning1
    _, s, t, polars, energy_pulse2 = am_twocolor_fortran(t_factor=t_factor,tau1=tau1, tau2=tau2, dt=dt, area1=area1, area2=area2, detuning=detuning1, t02=t02, detuning2=detuning2, rf_energy=rf_energy)
    e_pulse1 = detuning1# - rf_energy
    e_pulse2 = energy_pulse2# - rf_energy
    p1 = ps.Pulse(tau=tau1, e_start=e_pulse1, w_gain=0, e0=area1, t0=0)
    p2 = ps.Pulse(tau=tau2, e_start=e_pulse2, w_gain=0, e0=area2, t0=t02)
    pulse_total = ps.MultiPulse(p1, p2)
    print("E1:{:.4f}, E2:{:.4f}".format(detuning1,energy_pulse2))
    states = np.empty([len(t),2])
    states[:,0] = 1-s
    states[:,1] = s
    
    # t,x,pulse_total = am_twocolor(tau1, tau2, dt, area1, area2, detuning1, t02)
    # states = np.empty([len(t),2])
    # states[:,0] = 1-x[:,0].real
    # states[:,1] = x[:,0].real
    # polars = x[:,1]

    plt.plot(t*1e-3,states[:,0],label="g")
    plt.plot(t*1e-3,states[:,1],label="x")
    plt.xlabel("occupation")
    plt.ylabel("t (ps)")
    plt.legend()
    plt.show()
    e_values = np.empty([len(t),2])
    e_vectors = np.empty([len(t),2,2], dtype=complex)
    # eigh: returns w,v
    # w: Eigenvalues in ascending order
    # v: column [:,i] is the normalized eigenvector corresponding to the eigenvalue w[i]
    # this means e_vectors[t,:,i] is the normalized EVec to EV i at time t
    for i in range(len(t)):
        e_values[i],e_vectors[i] = np.linalg.eigh(h_twols(t[i],p1, detuning1/HBAR))
    
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
            print("{} center: ({:.2f},{:.2f})".format(i, *np.abs(e_vectors[int(len(t)/2),:,i])**2))

    
    for i in range(2):
        plt.plot(t*1e-3,e_values[:,i].real,label="E_{}".format(i))
    plt.plot(t*1e-3,np.abs((e_values[:,0]-e_values[:,1]).real),label="diff_E")
    plt.legend()
    plt.xlabel("t (ps)")
    plt.ylabel("E (meV)")
    plt.show()

    n_colors = np.empty([2,e_values.shape[0]])  # for gnuplot
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
    plt.xlabel("t (ps)")
    plt.ylabel("E (meV)")
    plt.show()

    dressed_states = np.empty([len(t),2])
    for j in range(len(t)):
        for i in range(2):
            dressed_states[j,i] = np.dot(np.abs(e_vectors[j,:,i])**2, states[j])
            dressed_states[j,i] += 2*np.real(e_vectors[j,0,i]*np.conj(e_vectors[j,1,i])*polars[j])
    
    indices = [[0,1],[0,2],
               [1,2]]
    dressed_states = np.empty([len(t), 2])
    for j in range(len(t)):
        for i in range(2):
            dressed_states[j,i] = np.dot(np.abs(e_vectors[j,:,i])**2,states[j])
            for k in range(1):
                l = indices[k][0]
                m = indices[k][1]
                dressed_states[j,i] += 2*np.real(e_vectors[j,l,i]*np.conj(e_vectors[j,m,i])*polars[j])
    
    for i in range(2):
        plt.plot(t*1e-3,dressed_states[:,i].real,label="state_{}".format(i))
    plt.plot(t*1e-3,states[:,1],label="x")
    plt.xlabel("t (ps)")
    plt.ylabel("occupation (dressed state)")
    plt.legend()
    plt.show()

    for i in range(2):
        plt.scatter(t*1e-3,dressed_states[:,i].real,c=s_colors[i])
    plt.xlabel("t (ps)")
    plt.ylabel("occupation (dressed state)")
    plt.show()

    if filename is not None:
        # for the pulse-less dressed states
        pulse_1 = ps.ChirpedPulse(tau_0=tau1, e_start=detuning1, alpha=0, t0=0, e0=0)
        pulse_2 = ps.Pulse(tau=tau2, e_start=detuning2, t0=t02, e0=0, phase=0)
        pulse = ps.MultiPulse(pulse_1, pulse_2)
        e_values_nopulse = np.empty([len(t), 2])
        for t_,i in zip(t,range(len(t))):
            e_values_nopulse[i],_ = np.linalg.eigh(h_twols(t_, pulse_1, detuning1/HBAR))

        e,c,s = e_values, n_colors, np.clip(dressed_states,0,1)
        export_csv(filename,t/1000,*np.transpose(e),*c,*np.transpose(s), states[:,0],states[:,1],*np.transpose(e_values_nopulse),np.abs(e_vectors[:,0,0])**2,np.abs(e_vectors[:,1,0])**2)

    # now analyse the action of the second pulse:
    # calculate the interaction hamiltonian belonging to the second pulse in the dressed state basis
    e_values_new = np.empty_like(e_values)
    h_interaction = np.empty([len(t),2,2],dtype=complex)
    # for i in tqdm.trange(len(t)):
    #     h_int = -np.array([[0,0.5*HBAR*p2.get_envelope(t[i])*np.exp(1j*(1/HBAR)*(detuning2-detuning1)*t[i])],
    #                       [0.5*HBAR*p2.get_envelope(t[i])*np.exp(-1j*(1/HBAR)*(detuning2-detuning1)*t[i]),0]])
    #     for j in range(2):
    #         for k in range(2):
    #             h_interaction[i,j,k] = np.dot(np.conjugate(np.transpose(e_vectors[i,:,j])),np.dot(h_int,e_vectors[i,:,k]))
        # for both dressed states |+> and |->, also an energy contribution happens
    for i in range(len(t)):
        for j in range(2):
            e_values_new[i,j] = e_values[i,j] + h_interaction[i,j,j].real 
    for i in range(2):
        plt.plot(t*1e-3,e_values_new[:,i].real,label="E_{}".format(i))
        plt.plot(t*1e-3,e_values[:,i].real,label="old_E_{}".format(i))
    # plt.plot(t*1e-3,np.abs((e_values_new[:,0]-e_values_new[:,1]).real),label="diff_E")
    plt.legend()
    plt.xlabel("t (ps)")
    plt.ylabel("E (meV)")
    plt.show()

    plt.plot(t*1e-3,(e_values_new[:,1]-e_values_new[:,0]).real,label="E_1-E_0")
    plt.plot(t*1e-3,(e_values[:,1]-e_values[:,0]).real,label="E_1-E_0, no pulse 2")
    plt.plot(t*1e-3,[np.abs(detuning1-detuning2) for _ in range(len(t))],label="delta_21")
    #p2_norm = p2.get_envelope(t) / np.max(p2.get_envelope(t))
    #plt.plot(t*1e-3,5*p2_norm,label="pulse")
    plt.legend()
    plt.xlabel("t (ps)")
    plt.ylabel("E (meV)")
    plt.show()

    if filename is not None:
        export_csv(filename.split(".")[0]+"_new_energies.csv",t*1e-3,e_values_new[:,0],e_values_new[:,1],(e_values_new[:,1]-e_values_new[:,0]))

    # def euler(x0, equation):
    #     # s = int((t1 - t0) / h)
    #     n_ = len(x0)
    #     h = np.abs(t[0]-t[1])
    #     # t = np.linspace(t0, t1, s + 1)
    #     x = np.zeros([len(t), n_], dtype=complex)
    #     x[0] = x0
    #     for i in tqdm.trange(len(t)-1):
    #         x[i + 1] = x[i] + h * equation(i, x[i])
    #     return t, np.array(x, dtype=complex)

    # def dressed_state_equation(i, x):
    #     plus = x[0]  #  |+><+|
    #     minus = x[1]  #  |-><-|
    #     polar = x[2]  # |+><-|

    #     b = h_interaction[i,1,0]  # interaction term * |+><-|
    #     e_plus = e_values_new[i,1]
    #     e_minus = e_values_new[i,0]

    #     _plus = 2/HBAR * np.imag(b*polar)
    #     _minus = -2/HBAR * np.imag(b*polar)
    #     _polar = -1j/HBAR*(e_minus-e_plus)*polar - (1j/HBAR) * (np.conjugate(b)*plus - b*minus)  #  * (2*np.real(b)*plus - b)
    #     return np.array([_plus, _minus, _polar], dtype=complex)
    
    # _, ds = euler(np.array([0,1,0]), dressed_state_equation)
    # plt.plot(t,ds[:,0].real,label="+")
    # plt.plot(t,ds[:,1].real,label="-")
    # plt.legend()
    # plt.ylim(-190,190)
    # plt.show()
    
    return t, e_values, n_colors, np.clip(dressed_states,0,1), np.abs(e_values[int(len(t)/2),1]-e_values[int(len(t)/2),0])



# twolevels_dressedstates(area2=0)
# twolevels_dressedstates(detuning1=0,area1=5*np.pi,tau2=100,area2=0,t02=0,detuning2=0)
# twolevels_dressedstates(detuning1=-8,area1=22.65*np.pi,tau1=2400,tau2=3040,area2=19.29*np.pi,t02=0,detuning2=-19.163)
# twolevels_dressedstates(tau1=2496, tau2=3585, area1=29.3*np.pi, area2=29.1*np.pi, t02=1945, detuning=-11.0)
