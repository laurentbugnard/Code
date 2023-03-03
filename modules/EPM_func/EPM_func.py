import numpy as np
from tqdm import tqdm
from GooseEPM import SystemAthermal
from scipy.signal import fftconvolve

def evolution(system:SystemAthermal, nstep: int) -> tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        system (SystemAthermal): The system to evolve.
        nstep (int): The number of steps.

    Returns:
        sigmabar (np.ndarray): List of mean stress.
        epspbar (np.ndarray): List of mean plastic strain.
    """

    sigmabar = np.empty([nstep + 1])  # average stress
    epspbar = np.empty([nstep + 1])  # average plastic strain
    sigmabar[0] = system.sigmabar
    epspbar[0] = np.mean(system.epsp)

    for i in range(1, nstep+1):
        system.shiftImposedShear()
        system.relaxAthermal()
        sigmabar[i] = system.sigmabar
        epspbar[i] = np.mean(system.epsp)

    return sigmabar,epspbar

def evolution_verbose(system:SystemAthermal, nsteps: int) -> dict:
    """Evolves the system by ``nstep`` and returns means and maps of ``sigma`` and ``epsp`` for each step. 
    Additionally, returns propagator, mean yield stress field, number of relaxation steps and indexes of first failing block for each avalanche.

    Args:
        system (SystemAthermal): The system to evolve.
        nstep (int): The number of steps.

    Returns:
        propagator (np.ndarray): Propagator.
        sigmay_mean (np.ndarray): Mean yield stress field.
        sigmabar (np.ndarray): List of mean stress.
        epspbar (np.ndarray): List of mean plastic strain.
        sigma (list): List of stress maps.
        epsp (list): List of plastic strain maps.
        relax_steps (np.ndarray): Number of relaxation steps for each avalanche.
        failing (np.ndarray): List of failing indexes.
    """
    sigmabar = np.empty([nsteps + 1])  # average stress
    sigmabar[0] = system.sigmabar
    
    epspbar = np.empty([nsteps + 1])  # average plastic strain
    epspbar[0] = np.mean(system.epsp)
    
    sigma = [system.sigma.copy()]
    
    epsp = [system.epsp.copy()]
    
    relax_steps = np.empty([nsteps + 1])
    relax_steps[0] = 0
    
    failing = np.empty([nsteps])
    
    print('Evolving system...')

    for i in tqdm(range(1, nsteps+1)):
        system.shiftImposedShear()
        relax_steps[i] = system.relaxAthermal()
        
        sigmabar[i] = system.sigmabar
        epspbar[i] = np.mean(system.epsp)
        sigma.append(system.sigma.copy())
        epsp.append(system.epsp.copy())
        failing[i-1] = np.argmax(np.abs(system.sigma) - system.sigmay) #TODO do it later instead (after evolution, before plotting --> when computing more things). Here, do only what is strictly necessary to extract. But then we need to extract sigmay at each step (or at least its changes).
        
    gammabar = sigmabar + epspbar
    
    print('Done.')
    
    return {'propagator': system.propagator,
            'sigmay_mean': system.sigmay_mean,
            'sigmabar': sigmabar,
            'epspbar': epspbar,
            'gammabar': gammabar,
            'sigma': sigma,
            'epsp': epsp,
            'relax_steps': relax_steps,
            'failing': failing}
        

def find_runtime_error(system:SystemAthermal, nstep:int, max_relaxationsteps = 100000) -> int:
    """Finds at which macrostep a runtime error occurs due to failure to relax with microsteps.
    Returns 0 if no error is found.

    Args:
        system (SystemAthermal): The system to evolve.
        nstep (int): The number of steps.
        max_relaxationsteps (int, optional): Maximum allowed relaxations steps. Defaults to 100000.

    Returns:
        int: Iteration at which the error occurs. If 0, no error occured.
    """

    for i in range(1, nstep+1):
        try:
            system.eventDrivenStep(max_steps = max_relaxationsteps)
        except:
            print(f'RuntimeError at step {i}')
            return i
    print('No error found')
    return 0

def init_sigma(system, sigma_std=0.1, seed=0):
    np.random.seed(seed)
    dsig = np.random.normal(0, sigma_std, system.shape)
    dsig_pad = np.pad(dsig,((0,system.shape[0]-1),(0,system.shape[1]-1)), mode='wrap')
    
    system.sigma = fftconvolve(dsig_pad, system.propagator, mode='valid')