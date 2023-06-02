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


def simulate(params, save=True, folder='./data', force_redo=False, full_processing=False, exact_nsteps=False):
    # In docstring: specify that nsteps is the number of shift+relax and that all outputs
    # will have length 2*nsteps + 1 (initial state)
    #TODO: by default, make it not return anything (a bit faster, no loading etc.)
    
    print('Parameters:', params, flush=True)
    
    #A) Format params and get the nsteps mode
    params = params.copy() #make sure to copy them to not modify
    params, nsteps_mode = format_params(params)
    
    #B) Check if simulation already exists
    if not force_redo:
        try:
            results = load_results(params, folder, nsteps_mode)
            return results
        except CustomException as e:
            print(e, flush=True)
        
        # except Exception as e:
        #     raise e
    
    else: print("Force redoing the simulation.", flush=True)
        
    #C) Create system (physics)
    system = create_system(params)
    
    #D) Initialize results (observer)
    results = Results(system, params['seed'], params['map_type'], params['sigma_std'])
    
    #E) Evolve system
    print('Evolving system...', flush=True)
    
    if nsteps_mode in ['fixed', 'min']:
        
        nsteps = params['nsteps']
        results._nsteps = nsteps
        
        for _ in tqdm(range(nsteps)):
            system.shiftImposedShear()
            results.add_observation(system)
            system.relaxAthermal()
            results.add_observation(system)
    
    elif nsteps_mode == 'min_flow':
    
        stop_condition = False
        progress = tqdm()
        i = 0
        
        while not stop_condition:
            system.shiftImposedShear()
            results.add_observation(system)
            system.relaxAthermal()
            results.add_observation(system)
            
            #use nsteps as imposed flow size
            results.decompose()
            stop_condition = ((results.idx_flow.size - 1)/2 >= params['nsteps'])
            
            progress.update()
            i +=1
        
        #set nsteps value to actual number of steps
        results._nsteps = i
        params['nsteps'] = i
        
    else: raise CustomException('Invalid nsteps mode.')
        
    
    #F) Process results
    print('Processing results...', flush=True)
    results.process_basic()
    
    if full_processing:
        results.process_stability()
        results.process_statistics()
        
    #G) Save results
    if save:
        print("Saving results...", flush=True)
        params['flow_samples'] = (results.idx_flow.size - 1)/2 #add this additional column before saving
        save_results(results, params, folder)
    
    
    return results


def format_params(params):
    #The goal of this function is to add some flexibility to the params. 
    #Also, it makes sure that every simulation is saved according to a convention
    
    
    # Make sure the number of decimals is max 5 (as a convention, to avoid precision problems)
    for key, val in params.items():
        try:
            params[key] = round(val,5)
        except:
            pass
    
    #Set some default values if not provided:
    if 'seed' not in params:
        params['seed'] = 0
    
    
    
    #FORMAT NSTEPS
    #If it is a tuple, extract the nsteps_mode:
    if type(params['nsteps']) == tuple:
        nsteps_mode = params['nsteps'][0]
        params['nsteps'] = params['nsteps'][1]
    
    #Otherwise use value as it is:
    else:
        nsteps_mode = 'fixed'
    
    return params, nsteps_mode

def load_results(params, folder, nsteps_mode):
    
    params_df = pd.DataFrame(params, index=[0])
    
    try:
        df = pd.read_csv(f'{folder}/results_df.csv', index_col=0)
        matches = find_matches(df, params_df, nsteps_mode)
        
        if len(matches) > 1:
            matches = matches[matches.nsteps == matches.nsteps.max()].iloc[0]
            warnings.warn(f"{len(matches)} simulations found. Using the one with max nsteps.")
        
        index = matches['index'].item()
    
    except FileNotFoundError:
        raise CustomException("No Simulation Dataframe found.")
    
    except ValueError:
        raise CustomException("No match found.")
    
    try:
        with open(f'{folder}/{index}.pkl', 'rb') as file:
            
            print('Loading:', df.loc[index].to_dict(), flush=True)
            
            return pickle.load(file)
    
    except FileNotFoundError:
        df = df.drop(index)
        df.to_csv(f'{folder}/results_df.csv')
        raise CustomException("Simulation exists but corrupted data. Removed row.")





def save_results(results, params, folder):
    
    params_df = pd.DataFrame(params, index=[0])
    
    try:
        df = pd.read_csv(f'{folder}/results_df.csv', index_col=0)
    
    except:
        df = pd.DataFrame(params, index=[0])
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
            raise CustomException("Invalid 'map_type'. Use 'homog, 'custom' or 'cg'.")        
    
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
def extract(params_list, func, folder='./data'):

    df = pd.DataFrame(columns=params_list[0])
    
    for params in tqdm(params_list):
        params, nsteps_mode = format_params(params)
        
        results = load_results(params, folder, nsteps_mode)
        params['nsteps'] = results._nsteps #make sure we're using the actual number of steps
        
        new_values_dict = func(results)
        new_row = pd.DataFrame(params, index=[0]).assign(**new_values_dict)
        
        df = df.merge(new_row, how='outer')
    
    return df

def find_matches(df, params_df, nsteps_mode, ignore=[]):
    
    #Ignoring certain keys
    for element in ignore:
        try:
            params_df = params_df.drop('element')
        except:
            warnings.warn(f"Could not ignore {element}, parameter doesn't exist.")
    
    match nsteps_mode:
        
        case 'fixed':
            pass #don't do anything
            
        case 'min':
            min_nsteps = params_df['nsteps'].iloc[0] #define the minimum number of steps necessary
            df = df.query(f"nsteps >= {min_nsteps}") #drop everything that has less total steps
            params_df = params_df.drop(columns='nsteps') #nsteps not relevant anymore to look for matches
            
        case 'min_flow':
            min_flow_samples = params_df['nsteps'].iloc[0] #define the minimum number of flow samples
            df = df.query(f"flow_samples >= {min_flow_samples}") #drop everything that has less flow samples
            params_df = params_df.drop(columns='nsteps') #nsteps not relevant anymore to look for matches

        case _:
            raise CustomException('Invalid nsteps mode.')
        
    return df.reset_index().merge(params_df) #trick to look for a match


class CustomException(Exception):
    pass