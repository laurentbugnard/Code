#%% IMPORTS
import matplotlib.pyplot as plt
import numpy as np
from GooseEPM import SystemAthermal
import h5py
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from ipy_config import*
ipy_config()

#%% LOAD DATA
f = h5py.File('data/data.hdf5','r')

propagator = np.array(f.get('propagators/propL=100')).real
propagator = propagator.copy()
L = propagator.shape[0]

sigmay_mean = np.array(f.get('sigmaY/L=1000beta=0.7xi=100000000.0p=0.1'))[0:100,0:100]

f.close()

#%% INITIALIZE
system_homog = SystemAthermal(
    propagator= propagator,
    distances_rows= np.fft.fftfreq(L)*L,
    distances_cols= np.fft.fftfreq(L)*L,
    sigmay_mean= np.ones((L,L)),
    sigmay_std= np.ones((L,L)) * 0.1,
    seed=123,
    failure_rate= 1
)

system_inhomog = SystemAthermal(
    propagator= propagator,
    distances_rows= np.fft.fftfreq(L)*L,
    distances_cols= np.fft.fftfreq(L)*L,
    sigmay_mean= sigmay_mean,
    sigmay_std= np.ones((L,L)) * 0.1,
    seed=123,
    failure_rate= 1
)

#%% EVOLUTION
nstep = 10

sigma_homog, epsp_homog = evolution(system_homog, nstep)
sigma_inhomog, epsp_inhomog = evolution(system_inhomog, nstep)


# %% PLOT EVOLUTION

fig = plt.figure(figsize = (15,12))
plt.subplot(2,1,1)
plt.plot(epsp_homog, sigma_homog, marker = '^', markersize = 5, color = 'blue', label = 'homogeneous')
plt.plot(epsp_inhomog, sigma_inhomog, markersize = 1, color = 'red', label = 'inhomogeneous')
plt.axhline(0,color = 'k')
plt.xlabel(r'$\epsilon_p$', fontsize = 20)
plt.ylabel(r'$\Sigma$', fontsize = 20)
plt.show()
plt.legend()


plt.subplot(2,2,3)
plt.imshow(sigmay_mean[0:100,0:100])
plt.colorbar()
plt.title(r'$\sigma^Y$ (mean)', fontsize = 20)
plt.subplot(2,2,4)
plt.imshow(propagator, norm = LogNorm())
plt.title('propagator', fontsize = 20)
plt.colorbar()
# %% ANIMATED

speed = 1
fig = plt.figure(figsize = (15,12))
graph, = plt.plot([], [], markersize = 1)
plt.xlim(np.min(epsp_homog), np.max(epsp_homog))
plt.ylim(np.min(sigma_homog), np.max(sigma_homog))
plt.axhline(0,color = 'k')
def animate(i):
    graph.set_data(epsp_homog[:speed*i], sigma_homog[:speed*i])
ani = FuncAnimation(fig, animate, frames=nstep, interval=1)


# %% STEP BY STEP EVOLUTION
system_sbs = SystemAthermal(
    propagator= propagator,
    distances_rows= np.fft.fftfreq(L)*L,
    distances_cols= np.fft.fftfreq(L)*L,
    sigmay_mean= np.ones((L,L)),
    sigmay_std= np.ones((L,L)) * 0.1,
    seed=123,
    failure_rate= 1
)

#%%
system_sbs.eventDrivenStep()

#%%
for i in range(100):
    system_sbs.eventDrivenStep()
plt.figure(figsize = (10,8))
plt.subplot(1,2,1)
plt.imshow(system_sbs.sigma)
plt.subplot(1,2,2)
plt.imshow(system_sbs.sigma > 1)

plt.title(f'max = {np.max(system_sbs.sigma)}')
# %% SMALL TEST
test_size = 30
propagator = np.zeros((test_size,test_size))
propagator[0,0] = -1
propagator[0,1] = 0.25
propagator[1,0] = 0.25
propagator[0,-1] = 0.25
propagator[-1,0] = 0.25

system_small = SystemAthermal(
    propagator = propagator,
    distances_rows = np.fft.fftfreq(test_size)*test_size,
    distances_cols = np.fft.fftfreq(test_size)*test_size,
    sigmay_mean = np.ones_like(propagator),
    sigmay_std = np.ones_like(propagator)*0.1,
    seed = 123
)

#%% evolution
sigma_small, epsp_small = evolution(system_small, 15, max_relaxation_steps=10000)

#%% step by step
# system_small.eventDrivenStep(max_steps = 10)
plt.figure(figsize = (10,8))
plt.subplot(1,2,1)
plt.imshow(system_small.sigma)
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(system_small.sigma > system_small.sigmay-0.1)

# %% PLOT EVOLUTION

fig = plt.figure(figsize = (15,12))
plt.subplot(2,1,1)
plt.plot(epsp_small, sigma_small, marker = '^', markersize = 5, color = 'blue', label = 'homogeneous')
plt.axhline(0,color = 'k')
plt.xlabel(r'$\epsilon_p$', fontsize = 20)
plt.ylabel(r'$\Sigma$', fontsize = 20)
plt.show()


# %%
