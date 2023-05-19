import os
os.chdir('C:/Users/laure/Documents/EPFL/MA/MA1/PCSL/EPM')
import sys
sys.path.append('./modules')

import numpy as np
import warnings
from datetime import datetime
from scipy.signal import fftconvolve
from tqdm import tqdm

from GooseEPM import elshelby_propagator
from GooseEPM import SystemAthermal

from Results.Results import Results
from CorrGen.CorrGen import CorrGen


def simulate(params, nsteps, save=True, mask=None, force_redo=False):
    #TODO: specify that nsteps is the number of shift+relax and that all outputs
    # will have length 2*nsteps + 1 (initial state)
    
    #A) Format params
    #TODO: maybe this should be done outside of here?
    params = format_params(params)
    
    #B) Check if simulation already exists
    #TODO: uncomment and implement
    # try:
    #     results = load_results(params)
    # except: #TODO: add different types of error (e.g. FileNotFound)
    #     print("No previous simulation.")
    # else:
    #     print("Returning previous simulation.")
    #     return results
        
    #C) Create system (physics)
    system = create_system(params)
    
    #D) Initialize results (observer)
    results = Results(system, nsteps, params['seed'], params['map_type'], params['sigma_std'], params['meta'])
    
    #E) Evolve system
    print('Evolving system...', flush=True)
    for _ in tqdm(range(nsteps)):
        system.shiftImposedShear()
        results.add_observation(system)
        system.relaxAthermal()
        results.add_observation(system)
    
    #F) Process results
    print('Processing results...', flush=True)
    results.process_basic()
    results.process_extra()
    #TODO: add other processing
        
        
    #G) Save results
    if save:
        #TODO
        print("Saving results...", flush=True)
        pass
    
    return results
    
    #TODO: delete everything
    #Give a standardized name to the simulation according to conventions
    name = to_str(params) + f'/seed={seed}'
    
    #check if file already exists
    if os.path.exists(file): mode = 'r+'
    else: mode = 'w'
    
    with h5py.File(file, mode=mode) as f:
        
        ### First, check if the simulation already exists ###
        if (name in f) and (not(force_redo)):
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
        
        
        
        #then save in file
        if save:
            
            print('Saving to file...')
            
            for k,v in res_dict.items():
                f.create_dataset(name + '/'+ k, data = v)
            
            #Store also datetime of creation
            f.create_dataset(name + '/date', 
                             data = datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
            #Store total number of states since initial state
            f.create_dataset(name + '/totalSteps', data = nsteps)
            
            print('Done saving. \n')
        return res_dict, CorrGen_params


def format_params(params):
    #TODO: format everything well (if things are correctable). Otherwise raise an error
    #The goal of this function is to add some flexibility to the params. 
    # For example if no seed was given, use seed 0
    return params

def load_results(params):
    #goes through the metadata and returns the simulation object
    #if it couldn't find the file: raise an FileNotFound error and delete from the metadata
    #TODO
    pass

def create_system(params):
    
    match params['map_type']:
        
        case 'homog':
            sigmay_mean = np.ones((params['L'], params['L']))
        
        case 'custom':
            image = np.load(params['path'])
            sigmay_mean = image_to_map(image, params)
        
        case 'cg':
            print("Generating correlations...", flush=True)
            cg = CorrGen(L=params['L'], xi=params['xi'])
            cg.generate_fields(method=params['method'], exponent=params['exponent'], seed=params['seed'])
            cg.generate_sigmaY(p=params['p'])
            sigmay_mean = cg.sigmaY
        
        case _ :
            #TODO: raise the correct type of error
            pass
        
    
    print('Initializing system...', flush=True)
    system = SystemAthermal(
        *elshelby_propagator(L=sigmay_mean.shape[0]),
        sigmay_mean=sigmay_mean,
        sigmay_std=params['sigmay_std'] * np.ones_like(sigmay_mean),
        seed=params['seed'],
        init_random_stress=False
    )
    init_sigma(system, sigma_std=params['sigma_std'], seed=params['seed'])
    
    return system

def image_to_map(image, params):

    print("Preparing map...", flush=True)
    
    #try to shift using parameters "low" and "high"
    try:
        assert np.nanmin(image) == 0 , "Image must have min = 0."
        assert np.nanmax(image) == 1 , "Image must have max = 1."
        sigmay_mean = (params['high'] - params['low']) * image + params['low']
    
    except:
        #try to shift using "shift" and "scale"
        try:
            sigmay_mean = params['shift'] + params['scale']*image
        #last possibility: just use image
        except:
            print()
            warnings.warn("Couldn't use any map shifting method: using raw image.")
            sigmay_mean = image
    
    sigmay_mean = np.nan_to_num(sigmay_mean, nan=float('inf'))
    
    return sigmay_mean
    
    
def init_sigma(system, sigma_std=0.1, seed=0, relax=True):
    np.random.seed(seed)
    dsig = np.random.normal(0, sigma_std, system.shape)
    dsig_pad = np.pad(dsig,((0,system.shape[0]-1),(0,system.shape[1]-1)), mode='wrap')
    
    system.sigma = fftconvolve(dsig_pad, system.propagator, mode='valid')
    
    if relax: system.relaxAthermal()
