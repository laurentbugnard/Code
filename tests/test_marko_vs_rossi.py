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
ipy_config()

#%% EVOLUTION FUNCTION
def evolution(system, nstep, max_relaxation_steps = 100000):
    sigma = np.empty([nstep])  # average stress
    epsp = np.empty([nstep])  # average plastic strain
    sigma[0] = system.sigmabar
    epsp[0] = np.mean(system.epsp)

    for i in range(1, nstep+1):
        system.eventDrivenStep(max_steps = max_relaxation_steps)
        sigma[i] = system.sigmabar
        epsp[i] = np.mean(system.epsp)

    if(np.sum((np.diff(epsp) < 0)) > 0):
        print('Warning: epsp not monotonic!')
    return sigma,epsp

# %% SMALL TEST
size = 20
f = h5py.File('../data/data.hdf5','r')

propagator_marko = np.array(f.get(f'propagators/marko_propL={size}')).real
propagator_marko = propagator_marko.copy()
L = propagator_marko.shape[0]

propagator_rossi = np.array(f.get(f'propagators/rossi_propL={size}')).real
propagator_rossi = propagator_rossi.copy()
L = propagator_rossi.shape[0]

f.close()

system_marko = SystemAthermal(
    propagator = propagator_marko,
    distances_rows = np.fft.fftfreq(L)*L,
    distances_cols = np.fft.fftfreq(L)*L,
    sigmay_mean = np.ones_like(propagator_marko),
    sigmay_std = np.ones_like(propagator_marko)*0.1,
    seed = 123
)

system_rossi = SystemAthermal(
    propagator = propagator_rossi,
    distances_rows = np.fft.fftfreq(L)*L,
    distances_cols = np.fft.fftfreq(L)*L,
    sigmay_mean = np.ones_like(propagator_rossi),
    sigmay_std = np.ones_like(propagator_rossi)*0.1,
    seed = 123
)

#%% TOTAL EVOLUTION
sigma_marko, epsp_marko = evolution(system_marko, 1000, max_relaxation_steps=10000)

sigma_rossi, epsp_rossi = evolution(system_rossi, 1000, max_relaxation_steps=10000)

# %% PLOT STRESS STRAIN

fig = plt.figure(figsize = (15,12))

plt.subplot(2,1,1)
plt.title('Marko', fontsize = 20)
plt.plot(epsp_marko, sigma_marko, marker = '^', markersize = 5, color = 'blue')
plt.axhline(0,color = 'k')
plt.xlabel(r'$\epsilon_p$', fontsize = 20)
plt.ylabel(r'$\Sigma$', fontsize = 20)

plt.subplot(2,1,2)
plt.title('Rossi', fontsize = 20)
plt.plot(epsp_rossi, sigma_rossi, marker = '^', markersize = 5, color = 'blue')
plt.axhline(0,color = 'k')
plt.xlabel(r'$\epsilon_p$', fontsize = 20)
plt.ylabel(r'$\Sigma$', fontsize = 20)

fig.tight_layout()
plt.show()
# %%
