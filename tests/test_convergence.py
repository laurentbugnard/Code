#%% IMPORTS
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
from GooseEPM import SystemAthermal
import h5py
from matplotlib.animation import FuncAnimation
from ipy_config import*
from matplotlib.colors import LogNorm
from evolution import *
ipy_config()


# %% SMALL TEST
f = h5py.File('../data/data.hdf5','r')
propagator = np.array(f.get('propagators/propL=100')).real
propagator = propagator.copy()
L = propagator.shape[0]

f.close()

system_small = SystemAthermal(
    propagator = propagator,
    distances_rows = np.fft.fftfreq(L)*L,
    distances_cols = np.fft.fftfreq(L)*L,
    sigmay_mean = np.ones_like(propagator),
    sigmay_std = np.ones_like(propagator)*0.1,
    seed = 123
)

system_small.sigma *= 0.1

#%% STEP BY STEP (evenDrivenStep)
system_small.eventDrivenStep(max_steps = 10000) #FAILS FOR L=100
# plt.figure(figsize = (10,8))
# plt.subplot(1,2,1)
# plt.imshow(system_small.sigma)
# plt.colorbar()
# plt.subplot(1,2,2)
# plt.imshow(system_small.sigma > system_small.sigmay-0.1)
#%% TOTAL EVOLUTION

sigma_small, epsp_small = evolution(system_small, 23, max_relaxation_steps=10000)

#%% STEP BY STEP (even avalanches)
# # ShiftImposedShear first

# plt.figure(figsize = (10,8))
# plt.subplot(1,2,1)
# plt.imshow(system_small.sigma, vmin=-1, vmax=1)
# plt.colorbar()

# old_sigma = system_small.sigmabar
system_small.shiftImposedShear()
# dsig = system_small.sigmabar-old_sigma

# plt.subplot(1,2,2)
# plt.title(f"Shifted by {dsig}")
# plt.imshow(system_small.sigma, vmin=-1, vmax=1)
# plt.colorbar()

# n_failures = system_small.eventDrivenStep()

# print('dsig = ',dsig, '\n n_failures = ',n_failures)

#%% ANALYSE RELAXATION
idx = system_small.makeWeakestFailureStep()

index = np.unravel_index(idx, propagator.shape)
plt.figure()
plt.title(idx, fontsize = 20)
plt.imshow(system_small.sigma)
plt.colorbar()


#%% suite
index = [0,0]
sigmas = []
sigbars = []

for i in range(100):
    system_small.makeWeakestFailureStep()
    sigmas.append(system_small.sigma[index[0],index[1]])
    sigbars.append(system_small.sigmabar)

#%% (suite)

#local sigma at unstability box
plt.figure()
plt.subplot(1,2,1)
plt.plot(sigmas, marker = '^')
plt.title(f'Local stress evolution in {index}')

plt.subplot(1,2,2)
plt.plot(sigbars)
plt.title('Relaxation')
    
    

# %% PLOT STRESS STRAIN

fig = plt.figure(figsize = (15,12))
plt.subplot(2,1,1)
plt.plot(epsp_small, sigma_small, marker = '^', markersize = 5, color = 'blue', label = 'homogeneous')
plt.axhline(0,color = 'k')
plt.xlabel(r'$\epsilon_p$', fontsize = 20)
plt.ylabel(r'$\Sigma$', fontsize = 20)
plt.show()
# %%
