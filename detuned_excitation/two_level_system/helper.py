import numpy as np
import matplotlib.pyplot as plt

HBAR = 6.582119514e2  # meV fs
c_light = 299.792  # nm/fs

def nm_to_mev(lambda_light):
    return HBAR * 2*np.pi*c_light / lambda_light

def mev_to_nm(energy_light):
    return HBAR * 2*np.pi*c_light / energy_light

def export_csv(filename, *arg, precision=4, delimit=','):
    """
    Exportiert Arrays als .csv Datei
    :param delimit: delimiter 
    :param filename: filename
    :param precision: number of decimal places after which the number is truncated
    :return: null
    """
    p = '%.{k}f'.format(k=precision)
    ps = []
    for arguments in arg:
        ps.append(p)
    try:
        np.savetxt(
            filename,
            np.c_[arg],
            fmt=ps,
            delimiter=delimit,
            newline='\n',
            # footer='end of file',
            # comments='# ',
            # header='X , MW'
        )
        print("[i] csv saved to {}".format(filename))
    except TypeError:
        print("TypeError occured")
        for arguments in arg:
            print(arguments)

def save2darray(filename, t, array, precision=4, delimit=','):
    """
    savs 2d numpy array as csv for gnuplot
    """
    shape = array.shape
    dim = shape[-1]
    v = []
    for i in range(dim):
        v.append(array[:,i])
    try:
        if len(t) == shape[0]:
            v.append(t)
        else:
            print("[i] no time axis")
    except:
        print("[i] t not an array")
    export_csv(filename, *v, precision=precision, delimit=delimit)

def save_colormap(filename, x ,y , *arg, precision=4, delimit=','):
    """
    z has to be dim (ylen, xlen)
    """
    z = []
    for arguments in arg:
        z.append(arguments.reshape(-1,order='F'))
    new_x = np.empty([len(x)*len(y)])
    new_y = np.empty_like(new_x)
    for i in range(len(x)):
        for j in range(len(y)):
            new_x[i*len(y)+j] = x[i]
            new_y[i*len(y)+j] = y[j]
    export_csv(filename, new_x,new_y,*z,precision=precision,delimit=delimit)

def plot_result(t, x=None, s=None):
    if x is None and s is None:
        print("you did not provide data")
        return 1
    data = 0
    if x is not None:
        data = x[:,0].real
    if s  is not None:
        data = s
    plt.xlabel("t")
    plt.ylabel("occupation")
    plt.plot(t,data)
    plt.show()

HBAR = 6.582119514e-1*1e3  # meV fs
g_ex = -0.65  # in plane electron g factor
g_ez = -0.8  # out of plane electron g factor
g_hx = -0.35  # in plane hole g factor
g_hz = -2.2  # out of plane hole g factor
d0 = 0.25 #250e-3  # meV exchange splittings
d1 = 0.12 #120e-3  # meV
d2 = 0.05 #5e-3  # meV
#delta_B = -d0  # biexciton binding
delta_E = 0.0  # meV
# mu_b = e_charge/(2*m_0) * HBAR  # C*eV*fs/kg
# mu_b = 9.2740100783e-24  # J/T
# mu_b = mu_b / e_charge * 1e3  # meV/T
mu_b = 5.7882818012e-2   # meV/T

def energies(bz, delta_B=-d0, delta_E=0.0, d0=0.25):
    E_dp = delta_E - d0/2 - mu_b*bz/2 * (3*g_hz + g_ez)
    E_bp = delta_E + d0/2 + mu_b*bz/2 * (-3*g_hz + g_ez)
    E_bm = delta_E + d0/2 - mu_b*bz/2 * (-3*g_hz + g_ez)
    E_dm = delta_E - d0/2 + mu_b*bz/2 * (3*g_hz + g_ez)
    E_b  = 2*delta_E - delta_B
    return E_dp, E_bp, E_bm, E_dm, E_b