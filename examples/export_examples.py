#%%
import sys
sys.path.append('../') #include parent folder in the path
import numpy as np
from GooseEPM import elshelby_propagator
from GooseEPM import SystemAthermal
from EPM_func import *
import h5py

#%%
nsteps = 1000
L = 100

f = h5py.File('../data/maps.hdf5','r+')
sigmay_mean_dict = {}
for alpha in [0.6,0.7,0.8]:
    sigmay_mean_dict.update({str(alpha): np.array(f.get(f'sigmaY/L=100alpha={alpha}xi=10p=0.1'))})

f.close()

#%%

f = h5py.File('../data/sim_results.hdf5','r+')  

for alpha, sigmay_mean in sigmay_mean_dict.items():

    #Initialize
    system = SystemAthermal(
        elshelby_propagator(L=L),
        sigmay_mean=sigmay_mean,
        sigmay_std= 0.3 * np.ones([L, L]),
        seed=0,
        init_random_stress=False,
        init_relax=True,
        sigmabar=0
    )
    
    #Evolution
    sim_results = evolution_verbose(system, nsteps)    
    #Save results  
    name = f'/sim_results_alpha={alpha}'
    if(name in f):
            del f[name]
    for k,v in sim_results.items():
        f.create_dataset(name + '/' + k, data = v)
        
f.close()
# %%
