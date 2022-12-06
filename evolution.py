import numpy as np


def evolution(system, nstep: int):

    sigmabar = np.empty([nstep + 1])  # average stress
    epspbar = np.empty([nstep + 1])  # average plastic strain
    sigmabar[0] = system.sigmabar
    epspbar[0] = np.mean(system.epsp)

    for i in range(1, nstep+1):
        system.shiftImposedShear()
        system.relaxAthermal()
        sigmabar[i] = system.sigmabar
        epspbar[i] = np.mean(system.epsp)

    if(np.sum((np.diff(epspbar) < 0)) > 0):
        print('Warning: epsp not monotonic!')
    return sigmabar,epspbar

def evolution_verbose(system, nstep: int):
    """Evolves the system by ``nstep`` and returns means and maps of ``sigma`` and ``epsp`` for each step. Additionally, returns number of relaxation steps and indexes of first failing block for each avalanche.

    Args:
        system (SystemAthermal): The system to evolve.
        nstep (int): The number of steps.

    Returns:
        sigmabar: List of mean stress.
        epspbar: List of mean plastic strain.
        sigma: List of stress maps.
        epsp: List of plastic strain maps.
        relax_steps: Number of relaxation steps for each avalanche.
        failing: List of failing indexes.
    """
    sigmabar = np.empty([nstep + 1])  # average stress
    sigmabar[0] = system.sigmabar
    
    epspbar = np.empty([nstep + 1])  # average plastic strain
    epspbar[0] = np.mean(system.epsp)
    
    sigma = [system.sigma]
    
    epsp = [system.epsp]
    
    relax_steps = np.empty([nstep + 1])
    relax_steps[0] = 0
    
    failing = np.empty([nstep])
    

    for i in range(1, nstep+1):
        system.shiftImposedShear()
        relax_steps[i] = system.relaxAthermal()
        
        sigmabar[i] = system.sigmabar
        epspbar[i] = np.mean(system.epsp)
        sigma.append(system.sigma)
        epsp.append(system.epsp)
        failing[i-1] = np.argmax(np.abs(system.sigma) - system.sigmay) #TODO do it later instead (after evolution, before plotting --> when computing more things). Here, do only what is strictly necessary to extract. But then we need to extract sigmay at each step (or at least its changes).
        
    return sigmabar, epspbar, sigma, epsp, relax_steps, failing
        

def find_runtime_error(system, nstep, max_relaxation_steps = 100000):

    for i in range(1, nstep+1):
        try:
            system.eventDrivenStep(max_steps = max_relaxation_steps)
        except:
            print(f'RuntimeError at step {i}')
            return i
    print('No error found')