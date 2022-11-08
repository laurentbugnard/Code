#%%
import matplotlib.pyplot as plt
import numpy as np
from GooseEPM import SystemAthermal
import h5py


#%%

f = h5py.File('data.hdf5','r')

system = SystemAthermal(
    propagator= f.get('propagators/propL=1000'),
    distances_rows=...,
    distances_cols=...,
    sigmay_mean= f.get('sigmaY/L=1000beta=0.7xi=100000000.0p=0.1'),
    sigmay_std= np.ones((1000,1000))*0.1,
    seed=123,
    failure_rate=...,
)

f.close()

nstep = 1000
sigma = np.empty([nstep])  # average stress
epsp = np.empty([nstep])  # average plastic strain
sigma[0] = system.sigmabar
epsp[0] = np.mean(system.epsp)

for i in range(1, nstep):
    system.eventDrivenStep()
    sigma[i] = system.sigmabar
    epsp[i] = np.mean(system.epsp)

fig, ax = plt.subplots()
ax.plot(epsp, sigma)
plt.show()
