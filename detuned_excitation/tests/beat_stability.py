from beat import * 
import numpy as np
import matplotlib.pyplot as plt


# f, s, t = test_twopulse(dt=5, tau1=3500, tau2=3000, area1=20*np.pi, area2=10*np.pi)
# plt.plot(t,s)
# plt.ylim([-0.1,1.1])
# plt.show()
# f, s, t = test_twopulse(dt=5, tau1=3500, tau2=2495,t02=-252.5, area1=20*np.pi, area2=10*np.pi)
# plt.plot(t,s)
# plt.ylim([-0.1,1.1])
# plt.show()

# f, s, t = test_twopulse(dt=5, tau1=3700, tau2=5700,t02=700, area1=22.5*np.pi, area2=22*np.pi)
# f,s,t = test_twopulse(dt=1, tau1=5600, tau2=7600, area1=29.7613*np.pi, area2=24.3561*np.pi, t02=0)
# plt.plot(t,s)
# plt.ylim([-0.1,1.1])
# plt.show()

# f,s,t = test_twopulse(dt=1, tau1=6192, tau2=9583, area1=29.0*np.pi, area2=29.0*np.pi, t02=-1812)
# plt.plot(t,s)
# plt.ylim([-0.1,1.1])
# plt.show()

f,s,t,p = test_twopulse(dt=1, tau1=6200, tau2=9600, area1=29.0*np.pi, area2=29.0*np.pi, t02=-1800)
plt.plot(t,s)
plt.ylim([-0.1,1.1])
plt.xlabel("t in fs")
plt.ylabel("Besetzung")
plt.show()

dur1 = np.linspace(2000, 10000, 100)
area1 = np.linspace(0, 30*np.pi, 100)
test_stability_pulse1(dur1, area1, tau2=9580, area2=29*np.pi, t02=-1810, dt=5)
test_stability_pulse2(dur1, area1, tau1=6190, area1=29*np.pi, t02=-1810, dt=5)
t0_arr = np.linspace(-8000,8000, 100)
test_stability_t0(t0_arr, dt=1, tau1=6190, tau2=9580, area1=29.0*np.pi, area2=29.0*np.pi)
area1 = np.linspace(0, 40*np.pi, 100)
test_stability_area(area1, area1, tau1=6190, tau2=9580, detuning=-5, t02=-1810, dt=5)

# stability with regards to the frequency of the second pulse
factors = np.linspace(-1.5,1.5,1000)
e = np.empty([len(factors)])
for i in range(len(factors)):
    e[i],_,_,_ = test_twopulse(factor=factors[i], dt=5, tau1=6200, tau2=9600,t02=-1800, area1=29*np.pi, area2=29*np.pi)
ind = np.unravel_index(np.argmax(e, axis=None), e.shape)
print("{}, f:{}".format(ind,factors[ind]))
plt.plot(factors,e)
plt.xlabel("a: E_2 = d - a * rabifreq * h")
plt.ylabel("f")
plt.show()