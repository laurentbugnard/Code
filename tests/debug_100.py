#%% IMPORTS
import sys
sys.path.append('../')
import matplotlib.pyplot as plt
import numpy as np
from GooseEPM import SystemAthermal
import h5py
from ipy_config import*
ipy_config()

def evolution(system, nstep, max_relaxation_steps = 100000):
    sigma = np.empty([nstep])  # average stress
    epsp = np.empty([nstep])  # average plastic strain
    sigma[0] = system.sigmabar
    epsp[0] = np.mean(system.epsp)

    for i in range(1, nstep):
        system.eventDrivenStep(max_steps = max_relaxation_steps)
        sigma[i] = system.sigmabar
        epsp[i] = np.mean(system.epsp)

    if(np.sum((np.diff(epsp) < 0)) > 0):
        print('Warning: epsp not monotonic!')
    return sigma,epsp


# %% Import and initialize size 100
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

####FIRST MAKE SIGMAS LESS BIG#####

system_small.sigma *= 0.1

#%% DO 23 steps (no problem) --> then check 24ths 1by1
sigma_small, epsp_small = evolution(system_small, 23, max_relaxation_steps=10000)

#%%
system_small.shiftImposedShear()

sigmas = []
sigbars = []
unstable = np.where(np.abs(np.ravel(system_small.sigma)) > np.ravel(system_small.sigmay))[0]

max_steps = 100
i = 0
fig = plt.figure()

unstable_sizeS = []
max_instab_list = []

while(unstable.size > 0 and i < max_steps):
    system_small.makeWeakestFailureStep()
    unstable = np.where(np.abs(np.ravel(system_small.sigma)) > np.ravel(system_small.sigmay))[0]
    
    unstable_sizeS.append(unstable.size)
    max_instab = np.max(np.abs(system_small.sigma) - system_small.sigmay)
    max_instab_list.append(max_instab)
    
    
    
    if(i%10 == 0):
        plt.subplot(3, 4 , int(np.ceil((i+1)/10)))
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
