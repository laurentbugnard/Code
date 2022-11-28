#%% IMPORTS
import numpy as np
from GooseEPM import SystemAthermal

#%% LOAD DATA
propagator = np.array([[1/2,0],
                       [-1,1/2]])
L = propagator.shape[0]
# TODO have a propagator that is 0 in 0


#%% INITIALIZE and EVOLUTION
system = SystemAthermal(
    propagator= propagator,
    distances_rows= np.fft.fftfreq(L)*L,
    distances_cols= np.flip(np.fft.fftfreq(L)*L),
    sigmay_mean= np.ones((L,L)),
    sigmay_std= np.ones((L,L)) * 0.0,
    seed=123,
    failure_rate= 1
)


#%% CHoose sigma
sigma = np.array([[0,0],
                  [0.1,0]])

#%% USING C++

system.sigma = sigma.copy()

#%%

system.eventDrivenStep()

print('using C++ \n',system.sigma)

#%% USING PYTHON
failure_shift = np.min(1-sigma)
sigma += failure_shift
sigma += propagator*1

imposed_mean = np.mean(sigma) - 1/L**2

sigma = sigma - np.mean(sigma) + imposed_mean

print('using python \n',sigma)

#%% test if difference is just a shift
print('difference \n',sigma-system.sigma)

# %%
