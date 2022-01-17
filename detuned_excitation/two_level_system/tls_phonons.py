from numpy.core.fromnumeric import transpose
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch._C import dtype
import tqdm

# constants
hq = 0.6582173  # hbar in meV*ps
pi = np.pi
kb = 0.08617  # kboltz in meV/K
i = 1j  # imaginary unit

# GaAs Parameters
rho = 33517.04251  # density in in meV*ps**2/nm**5, corresponds to 5.37 g/cm**3
De = 7000  # deformation potential of conduction band in meV
Dh = -3500  # deformation potential of valence band in meV
uLA = 5.114966   # speed of sound in nm/ps

def get_simul_duration(tau1, tau2, t02, t_continue):
    # calculate the time that has to be simulated, consider if one of the times is given as array
    # always the longest time is needed
    if isinstance(tau1,np.ndarray):
        tau1 = np.max(tau1)
    if isinstance(tau2,np.ndarray):
        tau2 = np.max(tau2)
    # t02 can be positive or negative
    if isinstance(t02,np.ndarray):
        t02 = np.max(np.abs(t02))
    
    t_max = 3.5*tau1 + t_continue
    t_min = -3.5*tau1
    if (tau2 + np.abs(t02)) > tau1:
        t_max = 3.5*(tau2 + np.abs(t02)) + t_continue
        t_min = -3.5*(tau2 + np.abs(t02))
    return t_max, t_min

def prepare_e_field(tau1, tau2, t02, area1, area2, detuning1, detuning2):
    # ugly but works so far
    n_arr = 0
    batch = 0
    if isinstance(tau1,np.ndarray):
        print("tau1 specified as array")
        batch = len(tau1)
        tau1 = tau1[:,np.newaxis]
        n_arr += 1
    if isinstance(tau2,np.ndarray):
        print("tau2 specified as array")
        batch = len(tau2)
        tau2 = tau2[:,np.newaxis]
        n_arr += 1
    if isinstance(t02,np.ndarray):
        print("t02 specified as array")
        batch = len(t02)
        t02 = t02[:,np.newaxis]
        n_arr += 1
    if isinstance(area1,np.ndarray):
        print("area1 specified as array")
        batch = len(area1)
        area1 = area1[:,np.newaxis]
        n_arr += 1
    if isinstance(area2,np.ndarray):
        print("area2 specified as array")
        batch = len(area2)
        area2 = area2[:,np.newaxis]
        n_arr += 1
    if isinstance(detuning1,np.ndarray):
        print("detuning1 specified as array")
        batch = len(detuning1)
        detuning1 = detuning1[:,np.newaxis]
        n_arr += 1
    if isinstance(detuning2,np.ndarray):
        print("detuning2 specified as array")
        batch = len(detuning2)
        detuning2 = detuning2[:,np.newaxis]
        n_arr += 1
    if n_arr == 0:
        print("no parameter specified as array.")
        # just convert one parameter to array so the numeric stuff works
        batch = 1
        tau1 = np.array([tau1])
        tau1 = tau1[:,np.newaxis]
    if n_arr > 1:
        print("more than one parameter specified as array, choose only one.")
        exit(1)
    return tau1, tau2, t02, area1, area2, detuning1, detuning2, batch
    

def tls_phonons(tau1, tau2, t02, area1, area2, detuning1, detuning2, timestep=0.003, phonons=True, phonon_scaling=1.0, temp=0.0, qd_width=4.0, num_q=150, t_continue=0.0, device='cpu'):
    """
    times are in picoseconds, areas in units of pi
    if phonons=False: num_q = 2, temp = 0.0, phonon_scaling = 0.0
    """
    device = torch.device(device)
    print("using device: {}".format(device))

    # simulation specific stuff
    q_max = 2.3
    # num_q = 150
    # timestep = 0.003  # in ps
    # pulse area batch
    # area1 = np.linspace(22.65*0.9,22.65*1.1,20, dtype=np.float32)
    # area1 = torch.from_numpy(area1)  #torch.tensor([1.0,2,3,4,5,6,7,8,9,10]) #22.65 #1.0  # pulse area / pi
    # temp = 4.0  # temperature in K
    ae = qd_width  # localisation length of electron
    # taunull = 2.4  # in ps
    # alpha = 0.0  # chirp

    if phonons == False:
        num_q = 2
        temp = 0.0
        phonon_scaling = 0.0
    
    # phonon_scaling = 1.0  # option to switch off phonons
    # detuning1 = -8.0 #-8.0  #0.0  # in meV
    # detuning2 = -19.163#-19.163  #0.0
    # tau2 = 3.04 #3.04  #1.0
    # area2 = 19.29 #0.0
    # t02 = -0.73 #0.0  # time delay of second pulse in ps
    # t_continue = 0.0
    
    # one of the pulse parameters can be batched, i.e., an array can be used instead of a single number.
    # we expect only one parameter of (tau1, tau2, t02, area1, area2, detuning1, detuning2) to be an array
    t_max, t_min = get_simul_duration(tau1, tau2, t02, t_continue)
    num_t = int((t_max-t_min)/timestep)
    dt = (t_max-t_min)/num_t  # calculate new temporal step dt


    tau1, tau2, t02, area1, area2, detuning1, detuning2, batch = prepare_e_field(tau1, tau2, t02, area1, area2, detuning1, detuning2)
    t = np.linspace(0,num_t,num_t,endpoint=False)*dt + t_min
    e_0_arr = np.empty([batch,len(t)], dtype=np.float32)
    e_0_arr = area1*np.sqrt(pi/2.0)/(tau1)*np.exp(-t**2/(2.0*tau1**2)) \
            + np.exp(-i*(detuning2-detuning1)/hq*(t-t02))*area2*np.sqrt(pi/2.0)/(tau2)*np.exp(-(t-t02)**2/(2.0*tau2**2))

    # variables
    f = torch.zeros([batch,3], dtype=torch.float32).to(device)
    g0 = torch.ones([batch,3], dtype=torch.float32).to(device)
    p = torch.zeros([batch,3], dtype=torch.complex64).to(device)

    B = torch.zeros([batch,3,num_q], dtype=torch.complex64).to(device)
    tp = torch.zeros_like(B, dtype=torch.complex64).to(device)
    tm = torch.zeros_like(B, dtype=torch.complex64).to(device)
    s = torch.zeros_like(B, dtype=torch.complex64).to(device)

    n_p = torch.zeros([batch,3,num_q,num_q],dtype=torch.complex64).to(device)
    n = torch.zeros_like(n_p).to(device)
    tpp = torch.zeros_like(n_p).to(device)
    tpm = torch.zeros_like(n_p).to(device)
    tmm = torch.zeros_like(n_p).to(device)
    sp = torch.zeros_like(n_p).to(device)
    sm = torch.zeros_like(n_p).to(device)

    # phonon stuff
    w = torch.zeros(num_q, dtype=torch.float32).to(device)  # phonon dispersion
    gkop = torch.zeros_like(w).to(device)  # carrier-phonon coupling 
    n_th = torch.zeros_like(w).to(device)  # thermal phonon occupation

    # helper variables (sums)
    sb = torch.zeros(batch,dtype=torch.float32).to(device)  # sum over B
    st = torch.zeros(batch,dtype=torch.float32).to(device)  # sum over t^+ + t^-, i.e., tp + tm
    stp = torch.zeros([batch, num_q], dtype=torch.complex64).to(device)  # sum over T^+ + T, i.e., tpp + tpm
    stm = torch.zeros_like(stp).to(device)  # sum over T + T^-, i.e., tpm + tmm
    sn = torch.zeros_like(stp).to(device)  # sum over n + n^+
    snp = torch.zeros_like(stp).to(device)  # sum over n^+ + n


    # tau = tau1 #taunull  # torch.sqrt(alpha**2/taunull**2 +taunull**2)  # the chirp changes the pulse duration
    # a = alpha/(alpha**2 + taunull**4)  # chirp parameter

    dq = q_max / num_q  # q spacing    

    ah = 0.87*ae  # hole localization length

    # array of all k values, i.e., dq*[0,1,2,3,...,num_q]
    k = torch.from_numpy(np.linspace(0,num_q,num_q,endpoint=False)*dq).to(device)

    w = uLA * k
    gkop = torch.sqrt(k/(2.0*rho*hq*uLA))*(De*torch.exp(-(0.5*ae*k)**2) - Dh*torch.exp(-(0.5*ah*k)**2))
    gkop = phonon_scaling * gkop**2  # this is |g_q|^2 

    # phonon polaron shift: integral over |g_q|^2/w_q  
    #Delta_ph = 0.0
    #for i_k in range(1,num_q):
    #    Delta_ph += 1/(2*pi**2)*gkop[i_k]*k[i_k]**2*dq/w[i_k]

    Delta_ph = torch.sum(1/(2*pi**2)*gkop[1:]*k[1:]**2*dq/w[1:])

    print("phononscaling:{:.4f}, polaron shift:{:.8f}".format(phonon_scaling, Delta_ph.to('cpu')))

    # [1:] excludes the first item in the array, which fixes division by zero, since k[0]=0, but gkop[0]=0 as well
    n_th[0] = 0
    if temp > 0.0:
        n_th[1:] = 1/(torch.exp(hq*uLA*k[1:]/(kb*temp)) - 1) * gkop[1:]

    old = 0
    now = 1
    new = 2

    f_arr = torch.zeros([batch, num_t],dtype=torch.float32)

    # first time step
    e_0_arr = torch.from_numpy(e_0_arr).to(device)
    e_0 = e_0_arr[:,0]
    p[:,now] = -dt*(i/2.0*e_0)

    # initial condition: g = 1
    # g0[:] = 1

    for i_t in tqdm.trange(1,num_t):
        # dw = a*t[i_t]+ Delta_L/hq - Delta_ph  # for chirp
        # here: rotating frame with first detuning
        # also taking into account the polaron shift Delta_ph
        dw = -Delta_ph + detuning1/hq

        e_0 = e_0_arr[:,i_t]

        f[:,new]  = f[:,old]  + 2.0*dt*(-torch.imag(p[:,now]*torch.conj(e_0)))
        g0[:,new] = g0[:,old] + 2.0*dt*torch.imag(p[:,now]*torch.conj(e_0))
        #p[new]  = p[old]*torch.exp(-2.0*i*(-dw)*dt) + 2.0*dt*torch.exp(-i*(-dw)*dt)*(i/2.0*e_0*(f[now]-g0[now]))

        p[:,new]  = p[:,old]*torch.exp(-2.0*i*(-dw+2.0*sb)*dt) + 2.0*dt*torch.exp(-i*(-dw+2.0*sb)*dt)*(i/2.0*e_0*(f[:,now]-g0[:,now])-i*st)

        # k loop
        B[:,new] = B[:,old]*torch.exp(-2.0*i*w*dt) - 2.0*dt*torch.exp(-i*w*dt)*i*gkop*f[:,now,None]

        tp[:,new] = tp[:,old]*torch.exp(-2.0*i*(-dw+w+2.0*sb[:,None])*dt) + 2.0*dt*torch.exp(-i*(-dw+w+2.0*sb[:,None])*dt)*\
                    ( i/2.0*e_0[:,None]*s[:,now] -i*p[:,now,None]*(gkop*(1.0-f[:,now,None])+snp+n_th) - i*stp)

        tm[:,new] = tm[:,old]*torch.exp(-2.0*i*(-dw-w+2.0*sb[:,None])*dt) + 2.0*dt*torch.exp(-i*(-dw-w+2.0*sb[:,None])*dt)*\
                    ( i/2.0*e_0[:,None]*torch.conj(s[:,now])-i*p[:,now,None]*(gkop*f[:,now,None]+sn+n_th) - i*stm)

        s[:,new] = s[:,old]*torch.exp(-2.0*i*w*dt) + 2.0*dt*torch.exp(-i*w*dt)*\
                   ( -i/2.0*e_0[:,None]*torch.conj(tm[:,now]) + i/2.0*torch.conj(e_0[:,None])*tp[:,now] - i*gkop*f[:,now,None]*g0[:,now,None])  

        # q loop

        # n_p[k,q,...]
        # where everything with newaxis is k-indexed and every array without newaxis is q-indexed
        # i.e.,   w[:,np.newaxis]+w = w[i_k] + w[i_q]
        # gkop*s[now,:,np.newaxis] = gkop[i_q]*s[i_k,now]
        n_p[:,new] = n_p[:,old]*torch.exp(-2.0*i*(w[:,np.newaxis]+w)*dt) + 2.0*dt*torch.exp(-i*(w[:,np.newaxis]+w)*dt)*\
                 ( -i*(gkop*s[:,now,:,None]).transpose(1,2) -i*gkop*s[:,now,:,np.newaxis])

        n[:,new] = n[:,old]*torch.exp(2.0*i*(w[:,np.newaxis]-w)*dt) + 2.0*dt*torch.exp(i*(w[:,np.newaxis]-w)*dt)*\
                     ( i*(gkop*s[:,now,:,None]).transpose(1,2) - i*gkop*torch.conj(s[:,now,:,np.newaxis]))

        tpp_help = i*(gkop*(1.0-f[:,now,None,None])+snp[:,None]+n_th[None,:])*tp[:,now,:,np.newaxis]
        tpp[:,new] = tpp[:,old]*torch.exp(-2.0*i*(-dw+w[:,np.newaxis]+w+2.0*sb[:,None,None])*dt)\
                       + 2.0*dt*torch.exp(-i*(-dw+w[:,np.newaxis]+w+2.0*sb[:,None,None])*dt)*(i/2.0*e_0[:,None,None]*sp[:,now]\
                       - tpp_help - tpp_help.transpose(1,2) \
                       + i*((gkop*s[:,now,:,None]).transpose(1,2) + gkop*s[:,now,:,np.newaxis])*p[:,now,None,None] )  
                       # - i*(gkop*(1.0-f[:,now,None,None])+snp[:,None]+n_th[None,:])*tp[:,now,:,np.newaxis] \
                       # - (i*(gkop*(1.0-f[:,now,None,None])+snp[:,None]+n_th[None,:])*tp[:,now,:,np.newaxis]).transpose(1,2) \
                       # + i*((gkop*s[:,now,:,None]).transpose(1,2) + gkop*s[:,now,:,np.newaxis])*p[:,now,None,None] ) 


        tmm_help =  i*(gkop*f[:,now,None,None]+sn[:,None]+n_th[None,:])*tm[:,now,:,np.newaxis]
        tmm[:,new] = tmm[:,old]*torch.exp(-2.0*i*(-dw-w[:,np.newaxis]-w+2.0*sb[:,None,None])*dt) \
                       + 2.0*dt*torch.exp(-i*(-dw-w[:,np.newaxis]-w+2.0*sb[:,None,None])*dt)*(i/2.0*e_0[:,None,None]*torch.conj(sp[:,now]) \
                       - tmm_help - tmm_help.transpose(1,2) \
                       - i*(gkop*torch.conj(s[:,now,:,np.newaxis]) + (gkop*torch.conj(s[:,now,:,None])).transpose(1,2))*p[:,now,None,None] )
                      # i*(gkop*f[:,now,None,None]+sn[:,None]+n_th[None,:])*tm[:,now,:,np.newaxis] \
                      # (i*(gkop*f[:,now,None,None]+sn[:,None]+n_th[None,:])*tm[:,now,:,np.newaxis]).transpose(1,2) \
                      # - i*(gkop*torch.conj(s[:,now,:,np.newaxis]) + (gkop*torch.conj(s[:,now,:,None])).transpose(1,2))*p[:,now,None,None] )

        sp[:,new] = sp[:,old]*torch.exp(-2.0*i*(w[:,np.newaxis]+w)*dt) + 2.0*dt*torch.exp(-i*(w[:,np.newaxis]+w)*dt) \
                      * (-i/2.0*e_0[:,None,None]*torch.conj(tmm[:,now]) \
                      + i/2.0*torch.conj(e_0[:,None,None])*tpp[:,now] \
                      - i*(1.0-2.0*f[:,now,None,None])*((gkop*s[:,now,:,None]).transpose(1,2)+gkop*s[:,now,:,np.newaxis]))

        sm[:,new] = sm[:,old]*torch.exp(2.0*i*(w[:,np.newaxis]-w)*dt) + 2.0*dt*torch.exp(i*(w[:,np.newaxis]-w)*dt) \
                          * (-i/2.0*e_0[:,None,None]*torch.conj(tpm[:,now].transpose(1,2)) \
                          + i/2.0*torch.conj(e_0[:,None,None])*tpm[:,now] \
                          - i*(1.0-2.0*f[:,now,None,None])*(gkop*torch.conj(s[:,now,:,np.newaxis])-(gkop*s[:,now,:,None]).transpose(1,2)) )                         

        # look out in tpm: torch.conj(sm[now].transpose) should be sm[now] ? 
        # this was done because in the original code, only a triangular matrix was calculated for sm, but for tpm you need all indices
        # this leads to the same results
        # + 2.0*dt*torch.exp(-i*(-dw-w[:,np.newaxis]+w+2.0*sb[:,None,None])*dt)*(i/2.0*e_0[:,None,None]*torch.conj(sm[:,now].transpose(1,2)) \
        tpm[:,new] = tpm[:,old]*torch.exp(-2.0*i*(-dw-w[:,np.newaxis]+w+2.0*sb[:,None,None])*dt) \
                       + 2.0*dt*torch.exp(-i*(-dw-w[:,np.newaxis]+w+2.0*sb[:,None,None])*dt)*(i/2.0*e_0[:,None,None]*sm[:,now] \
                       - i*(gkop*(1.0-f[:,now,None,None])+snp[:,None]+n_th[None,:])*tm[:,now,:,np.newaxis] \
                       - (i*(gkop*f[:,now,None,None]+sn[:,None]+n_th[None,:])*tp[:,now,:,None]).transpose(1,2) \
                       + i*(gkop*torch.conj(s[:,now,:,None]) -(gkop*s[:,now,:,None]).transpose(1,2))*p[:,now,None,None] )

        # helper variables

        sb = torch.sum(1/(2*pi**2)*k**2*dq*torch.real(B[:,new]),axis=1)
        st = torch.sum(1/(2*pi**2)*k**2*dq*(tp[:,new]+tm[:,new]),axis=1)

        stp = torch.sum(1/(2*pi**2)*k**2*dq*(tpp[:,new]+tpm[:,new].transpose(1,2)),axis=2)
        stm = torch.sum(1/(2*pi**2)*k**2*dq*(tpm[:,new]+tmm[:,new]),axis=2)
        snp = torch.sum(1/(2*pi**2)*k**2*dq*(n_p[:,new]+torch.conj(n[:,new])),axis=2)
        sn = torch.sum(1/(2*pi**2)*k**2*dq*(n[:,new]+torch.conj(n_p[:,new])),axis=2)

        old = now
        now = new
        new = 3 - old - now

        f_arr[:,i_t] = f[:,now]
    return t, f_arr.to('cpu').numpy()

# area1s = np.linspace(1,2,2)
# t, f_arr = tls_phonons(tau1=2.4,tau2=1.0,t02=0.0,area1=area1s,area2=0.0,detuning1=0.0,detuning2=0.0,temp=77.0,device=torch.device("cuda"))
# print(f_arr[:,-1])
# plt.plot(t, f_arr[0,:])
# plt.savefig("phonons_77k.png")
# plt.plot(t,f_arr[0])
# plt.savefig("10pi.png")
