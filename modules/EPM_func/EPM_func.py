import numpy as np
from tqdm import tqdm
from GooseEPM import SystemAthermal
from scipy.signal import fftconvolve
from fastkde.fastKDE import pdf as pdf_kde


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
        sigmay (list): List of yield stress fields.
        sigma (list): List of stress maps.
        epsp (list): List of plastic strain maps.
        shift (np.ndarray): List of shifts before relaxation.
        relax_steps (np.ndarray): Number of relaxation steps for each avalanche.
        failing (np.ndarray): List of failing indexes.
    """
    sigmabar = np.empty([nsteps + 1])  # average stress
    sigmabar[0] = system.sigmabar
    
    epspbar = np.empty([nsteps + 1])  # average plastic strain
    epspbar[0] = np.mean(system.epsp)
    
    sigmay = [system.sigmay.copy()]
    
    sigma = [system.sigma.copy()]
    
    epsp = [system.epsp.copy()]
    
    shift = np.empty([nsteps + 1])
    shift[-1] = 0
    
    relax_steps = np.empty([nsteps + 1])
    relax_steps[0] = 0
    
    failing = np.empty([nsteps])
    
    print('Evolving system...')

    for i in tqdm(range(1, nsteps+1)):
        shift[i-1] = system.shiftImposedShear()
        relax_steps[i] = system.relaxAthermal()
        
        sigmabar[i] = system.sigmabar
        epspbar[i] = np.mean(system.epsp)
        sigmay.append(system.sigmay.copy())
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
            'sigmay': sigmay,
            'sigma': sigma,
            'epsp': epsp,
            'shift': shift,
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

def init_sigma(system, sigma_std=0.1, seed=0, relax=True):
    np.random.seed(seed)
    dsig = np.random.normal(0, sigma_std, system.shape)
    dsig_pad = np.pad(dsig,((0,system.shape[0]-1),(0,system.shape[1]-1)), mode='wrap')
    
    system.sigma = fftconvolve(dsig_pad, system.propagator, mode='valid')
    
    if relax: system.relaxAthermal()
    
    
    
def precompute(res_dict, mask=None):
    #Preprocessing: prepare histograms
    #STABILITY HISTOGRAM
    sigma = res_dict['sigma']
    sigmay = res_dict['sigmay']
    sigmay_mean = res_dict['sigmay_mean']
    #prepare bins
    max_non_inf = np.max(sigmay_mean[np.isfinite(sigmay_mean)])
    stability_bins_edges = np.linspace(0,2*max_non_inf, 20)
    #make a list of histograms (one for each step)
    stability_hist_list = []
    stability_kde_x_list = []
    stability_kde_y_list = []
    print("Precompute stability histograms and KDE...")
    for index in tqdm(range(len(sigma))):
        #x = stability
        if mask == None: x = sigmay[index] - sigma[index]
        else:
            x = sigmay[index][mask.astype('bool')] - sigma[index][mask.astype('bool')]

        n, _ = np.histogram(x, stability_bins_edges, density = True)
        kde_y, kde_x = pdf_kde(x.ravel()[x.ravel() < float('inf')], numPoints=257)
        stability_hist_list.append(n)
        stability_kde_x_list.append(kde_x.astype('float64'))
        stability_kde_y_list.append(kde_y.astype('float64'))
    
    
    #AVALANCHE SIZE HISTOGRAM
    relax_steps = res_dict['relax_steps']
    #prepare bins
    max_exponent = int(np.log10(np.max(relax_steps)))
    scales = np.logspace(0,max_exponent,max_exponent+1).astype('int')
    relax_steps_bins_edges = (np.array([1,2,5]) * scales.reshape(-1,1)).ravel() #broadcasting
    relax_steps_bins_edges = np.r_[relax_steps_bins_edges, 10*scales[-1]]
    #make a list of histograms (one for each step)
    relax_steps_hist_list = []
    print("Precompute avalanche histograms...")
    for index in tqdm(range(relax_steps.size)):
        n, _ = np.histogram(relax_steps[1:index+1], relax_steps_bins_edges)
        relax_steps_hist_list.append(n)
        
    res_dict.update({'stability_hist_list':stability_hist_list, 
                     'stability_kde_x_list':stability_kde_x_list, 
                     'stability_kde_y_list':stability_kde_y_list, 
                     'relax_steps_hist_list':relax_steps_hist_list,
                     'stability_bins_edges':stability_bins_edges,
                     'relax_steps_bins_edges':relax_steps_bins_edges,})

    return res_dict