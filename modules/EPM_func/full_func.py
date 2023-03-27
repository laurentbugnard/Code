#%%
import os
os.chdir('C:/Users/laure/Documents/EPFL/MA/MA1/PCSL/EPM')
import sys
sys.path.append('./modules')
from data_func.data_func import *
import numpy as np
from GooseEPM import elshelby_propagator
from GooseEPM import SystemAthermal
from EPM_func.EPM_func import *
import h5py
from CorrGen.CorrGen import CorrGen
from datetime import datetime

#%%

def full_simulation(params, nsteps, seed=0, save=True, file='./data/data.hdf5', mask=None):
    
    #Prepare CorrGen parameters for the output
    CorrGen_params=None
    if params['map_type'] == 'cg':
        CorrGen_params = {'xi': params['xi'], 'method': params['method'],
                            'exponent': params['exponent'], 'p': params['p']}
    
    #Give a standardized name to the simulation according to conventions
    name = to_str(params) + f'/seed={seed}'
    
    #check if file already exists
    if os.path.exists(file): mode = 'r+'
    else: mode = 'w'
    
    with h5py.File(file, mode=mode) as f:
        
        ### First, check if the simulation already exists ###
        if name in f:
            #Check the total number of steps
            totalSteps = f.get(name + "/totalSteps")[...]
            
            #If the simulation went far enough
            if nsteps <= totalSteps:
                date = f.get(name + '/date')[...]
                print(f'A simulation with these parameters and nsteps={totalSteps} already exists.')
                print(f'Returning previous results ({date}) truncated to nsteps={nsteps}.')
                
                #Get the previous data and turn it back into a dictionary
                res_group = f.get(name)
                res_dict = {}
                for key in res_group.keys():
                    #Only truncate values that have the size of totalSteps (or close)
                    try: #try block to avoid errors
                        if res_group[key].shape[0] == totalSteps + 1:
                            res_dict.update({key: res_group[key][...][:nsteps+1]})
                        elif res_group[key].shape[0] == totalSteps:
                            res_dict.update({key: res_group[key][...][:nsteps]})
                        else:
                            res_dict.update({key: res_group[key][...]}) #no truncation
                            
                    except: #still no truncation (execute if there was an error)
                        res_dict.update({key: res_group[key][...]}) #no truncation
                        
                    
                del res_dict['date']; del res_dict['totalSteps']
                    
                return res_dict, CorrGen_params
            
            #If the simulation was too short
            else:
                #Delete it and restart from 0
                print(f'A simulation with these parameters but a too small number of steps (nsteps={totalSteps}) already exists.')
                print('Deleting and starting from zero.')
                del f[name] #TODO: don't restart from 0, but from last state instead
        
        ### Otherwise, do the simulation ###
        
        if params['map_type'] =='homog':
            sigmay_mean = np.ones((params['L'], params['L']))
        elif params['map_type'] =='cg':
            cg = CorrGen(L=params['L'], xi=params['xi'])
            cg.generate_fields(method=params['method'], exponent=params['exponent'], seed=seed)
            cg.generate_sigmaY(p=params['p'])
            sigmay_mean = cg.sigmaY
        elif params['map_type'] == 'custom':
            image = np.load(params['path'])
            sigmay_mean = params['shift'] + params['scale']*image
            

        #Initialize
        print('Initializing system...', flush=True)
        system = SystemAthermal(
            *elshelby_propagator(L=sigmay_mean.shape[0]),
            sigmay_mean=sigmay_mean,
            sigmay_std=params['sigma_std'] * np.ones_like(sigmay_mean),
            seed=seed,
            init_random_stress=False
        )
        init_sigma(system, sigma_std=params['stability'], seed=seed)
        print('Done', flush=True)

        #Evolve
        res_dict = evolution_verbose(system, nsteps)
        
        #Precompute histograms and KDE's
        res_dict = precompute(res_dict, mask=mask)
        
        
        #then save in file
        if save:
            for k,v in res_dict.items():
                f.create_dataset(name + '/'+ k, data = v)
            
            #Store also datetime of creation
            f.create_dataset(name + '/date', 
                             data = datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
            #Store total number of states since initial state
            f.create_dataset(name + '/totalSteps', data = nsteps)
            
        return res_dict, CorrGen_params
    
#%%