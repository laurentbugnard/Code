#%%
import sys
sys.path.append('../') #include parent folder in the path
import numpy as np
from GooseEPM import elshelby_propagator
from GooseEPM import SystemAthermal
from evolution import *
import h5py

#%%
n_steps = 1000
L = 100
sigmay_mean=np.ones([L, L])
propagator, distances_rows, distances_cols = elshelby_propagator(L=L, imposed="strain")

f = h5py.File('./data/sim_results.hdf5','r+')  

for std in [0, 0.1, 0.3, 0.5]:

    #Initialize
    system = SystemAthermal(
        propagator=propagator,
        distances_rows=distances_rows,
        distances_cols=distances_cols,
        sigmay_mean=sigmay_mean,
        sigmay_std= std * np.ones([L, L]),
        seed=0,
        init_random_stress=False,
        init_relax=True,
        sigmabar=0
    )
    
    #Evolution
    sim_results = evolution_verbose(system, n_steps)
    sim_results.update({'sigmay_mean':sigmay_mean, 'propagator':propagator})
    
    #Save results  
    name = f'/sim_results_std={std}'
    if(name in f):
            del f[name]
    for k,v in sim_results.items():
        f.create_dataset(name + '/' + k, data = v)
        
f.close()
# %%
