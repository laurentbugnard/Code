import os
os.chdir('C:/Users/laure/Documents/EPFL/MA/MA1/PCSL/EPM')
import sys
sys.path.append('./modules')

import numpy as np
import warnings
from scipy.signal import fftconvolve
from tqdm import tqdm
import pandas as pd
import pickle

from GooseEPM import elshelby_propagator
from GooseEPM import SystemAthermal

from Results.Results import Results
from CorrGen.CorrGen import CorrGen


def simulate(params, save=True, folder='./data', force_redo=False, full_processing=True):
    # In docstring: specify that nsteps is the number of shift+relax and that all outputs
    # will have length 2*nsteps + 1 (initial state)
    #TODO: by default, make it not return anything (a bit faster, no loading etc.)
    
    print('Parameters:', params, flush=True)
    
    #A) Format params
    params, params_df = format_params(params)
    
    #B) Check if simulation already exists
    #TODO: consider using hydra instead
    if not force_redo:
        try:
            results = load_results(params_df, folder)
            print("Returning previous simulation.")
            return results
        except:
            print("No previous simulation.")
        
    #C) Create system (physics)
    system = create_system(params)
    
    #D) Initialize results (observer)
    results = Results(system, params['nsteps'], params['seed'], params['map_type'], params['sigma_std'])#, params['meta'])
    
    #E) Evolve system
    print('Evolving system...', flush=True)
    for _ in tqdm(range(params['nsteps'])):
        system.shiftImposedShear()
        results.add_observation(system)
        system.relaxAthermal()
        results.add_observation(system)
    
    #F) Process results
    print('Processing results...', flush=True)
    results.process_basic()
    if full_processing:
        results.process_stability()
        results.process_statistics()
        
    #G) Save results
    if save:
        print("Saving results...", flush=True)
        save_results(results, params_df, folder)
    
    
    return results


def format_params(params):
    #TODO: format everything well (if things are correctable). Otherwise raise an error
    #The goal of this function is to add some flexibility to the params. 
    # For example if no seed was given, use seed 0
    # Also, it makes sure that every simulation is saved according to a convention
    # Make sure the number of decimals is max 5 or something like this
    params_df = pd.DataFrame(params, index=[0])
    # params_df['meta'] = [params['meta']]
    
    return params, params_df

def load_results(params_df, folder):
    try:
        #load results_df
        df = pd.read_csv(f'{folder}/results_df.csv', index_col=0)
        
        #trick to look for a match. Raises an error if no or multiple matches.
        #TODO: check if it works in more complicated cases, e.g. when we have new columns (due to other map_types)
        index = df.reset_index().merge(params_df)['index'].item()
    
    except:
        raise Exception("Simulation doesn't exist.")
    
    else:
        try:
            with open(f'{folder}/{index}.pkl', 'rb') as file:
                return pickle.load(file)
        
        except:
            df = df.drop(index)
            df.to_csv(f'{folder}/results_df.csv')
            raise Exception("Simulation exists but corrupted data. Removed row.")

def save_results(results, params_df, folder):
    try:
        df = pd.read_csv(f'{folder}/results_df.csv', index_col=0)
    
    except:
        df = params_df
        new_index = df.index[0]
    
    else:
        #find the smallest new index
        new_index = np.max(df.index) + 1
        params_df.index = [new_index]
        #add row
        df = pd.concat([df, params_df])
    
    finally:
        #save df
        df.to_csv(f'{folder}/results_df.csv')
        
        #save Results object
        with open(f'{folder}/{new_index}.pkl', 'wb') as file:
                pickle.dump(results, file)

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
            raise Exception("Invalid 'map_type'. Use 'homog, 'custom' or 'cg'.")        
    
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


#TODO: very slow, try optimizing
def extract(params_list, func, ignore=[], folder='./data', exact_nsteps=False):
    
    #Ignoring certain keys
    for element in ignore:
        for params in params_list:
            try:
                del params[element]
            except:
                pass

    df = pd.DataFrame(columns=params_list[0])
    
    for params in tqdm(params_list):
        _, params_df = format_params(params)
        
        results = load_results(params_df, folder, exact_nsteps)
        
        #TODO: update also the nsteps if it was not exact
        new_values_dict = func(results)
        new_row = params_df.assign(**new_values_dict)
        
        df = df.merge(new_row, how='outer')
    
    return df