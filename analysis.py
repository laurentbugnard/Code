#%%
from plot_func import show_results
from GooseEPM import SystemAthermal
from GooseEPM import elshelby_propagator
import h5py
from CorrGen import *
from EPM_func import *
import sys
sys.path.append('../')
from config.ipy_config import ipy_config
ipy_config()
import copy
#%%
#Decide what to do in the document
create_maps = True
initialize = True
simulate = True
#%%
L = 128
xi = float('inf')
alpha_list = np.linspace(0,1.5,151)
p = 2

if create_maps:
    #Initialize empty maps dictionary
    sigmay_dict = {'homogeneous': np.ones((L,L))}
    for alpha in alpha_list:
        #Generate correlations
        corr = CorrGen(L, xi)
        corr.generate_fields(method='alpha',exponent=alpha)
        corr.generate_sigmaY(p)
        #append to dictionary
        sigmay_dict.update({str(alpha): corr.sigmaY.copy()})
    
    #Don't do it again
    # create_maps = False


#%%
#Initialize a system
if initialize:
    #choose sigmay_mean
    sigmay_mean = sigmay_dict['0.01']
    
    system = SystemAthermal(
        *elshelby_propagator(L),
        sigmay_mean=sigmay_mean,
        sigmay_std= 0.0*np.ones_like(sigmay_mean),
        seed=0,
        init_random_stress=True
    )
        
    initialize = False

#%%
nsteps = 1200

if simulate:

    #Make a copy to let system untouched
    evolving_system = copy.copy(system)

    #Change the system's initial stability
    evolving_system.sigma *= 2

    #Evolve
    results = evolution_verbose(evolving_system, nsteps)

#%%
show_results(**results)
# anim = show_results(**results, show_animation = True, fps=60, rate = 5)

# %%
