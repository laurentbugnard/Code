#%% IMPORTS
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
from GooseEPM import SystemAthermal
import h5py
from evolution import *
from ipy_config import*
ipy_config()


# %% Import and initialize size 100
method = 'rossi'
f = h5py.File('../data/data.hdf5','r')
propagator = np.array(f.get(f'propagators/{method}_propL=100')).real
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

#%% DO 62 steps (no problem) --> then check 63ths 1by1
#Marko: 63
#Rossi: 399
find_runtime_error(system_small, 500, 10000)
#%%
sigma_small, epsp_small = evolution(system_small, 212, max_relaxation_steps=10000)

state = system_small.state
sigma = system_small.sigma.copy()
epsp = system_small.epsp.copy()

#%% Go back to wanted system state

system_small = SystemAthermal(
    propagator = propagator,
    distances_rows = np.fft.fftfreq(L)*L,
    distances_cols = np.fft.fftfreq(L)*L,
    sigmay_mean = np.ones_like(propagator),
    sigmay_std = np.ones_like(propagator)*0.1,
    seed = 123,
    init_random_stress = False,
    init_relax = False
)

system_small.state = state
system_small.sigma = sigma
system_small.epsp = epsp

#%%
system_small.shiftImposedShear()

sigmas = []
sigbars = []
unstable = np.where(np.abs(np.ravel(system_small.sigma)) > np.ravel(system_small.sigmay))[0]

max_steps = 50000
i = 0
plot_period = max_steps/10
fig = plt.figure()

unstable_sizeS = []
max_instab_list = []

while(unstable.size > 0 and i < max_steps):
    system_small.makeWeakestFailureStep()
    unstable = np.where(np.abs(np.ravel(system_small.sigma)) > np.ravel(system_small.sigmay))[0]
    
    unstable_sizeS.append(unstable.size)
    max_instab = np.max(np.abs(system_small.sigma) - system_small.sigmay)
    max_instab_list.append(max_instab)
    
    
    
    if(i%plot_period == 0):
        plt.subplot(3, 4 , int(np.ceil((i+1)/plot_period)))
        plt.imshow(np.abs(system_small.sigma) > system_small.sigmay)
        plt.colorbar()
        plt.title(f'sigma_bar = {system_small.sigmabar:.2f} \n instab.max = {max_instab:.2f}, n_unst. = {unstable.size}')
        
        
    i+=1
    
plt.subplot(3,4, 11)
plt.plot(unstable_sizeS)
plt.title('Number of unstable particles')
plt.subplot(3,4, 12)
plt.title('Strongest instability')
plt.plot(max_instab_list)
print('n of steps: ',i)

fig.tight_layout()

#%% (suite)

#local sigma at unstability box
plt.figure()
plt.subplot(1,2,1)
plt.plot(sigmas, marker = '^')
plt.title('Stress of box that just broke')

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
