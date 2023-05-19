import numpy as np
from GooseEPM import SystemAthermal
from tqdm import tqdm
from fastkde.fastKDE import pdf as pdf_kde
from Plotting.plotting import power_law, power_law_fit

class Results(object):
    #parameters
    _L:int
    _nsteps:int
    _seed:int
    
    _map_type:str
    _sigmay_mean:np.ndarray
    _sigmay_std:np.ndarray
    
    _sigma_std:float
    
    #meta parameters
    _meta:dict
    
    #results
    epsp:list
    sigma:list
    sigmay:list
    
    epspbar:list #TODO: should I go back to using np.ndarray?
    sigmabar:list
    
    #processed results: #TODO: add the possible ones here?
    
    
    def __init__(self, system,
                 nsteps, seed, map_type, sigma_std,
                 meta = {}):
        assert system.shape[0] == system.shape[1], "System not square."
        assert type(nsteps) == int, "nsteps must be int."
        assert type(seed) == int, "seed must be int."
        assert type(map_type) == str, "map_type must be str."
        # assert type(sigma_std) == float, "sigma_std must be float." #TODO: see how to correct
        assert type(meta) == dict, "meta must be dict."
        
        #parameters
        self._L = system.shape[0]
        self._nsteps = nsteps
        self._seed = seed
        self._map_type = map_type
        self._sigmay_mean = system.sigmay_mean
        self._sigmay_std = system.sigmay_std
        self._sigma_std = sigma_std
        self._propagator = system.propagator
        
        #meta-parameters
        self._meta = meta
        
        #initialize results
        self.epsp = [system.epsp.copy()]
        self.sigma = [system.sigma.copy()]
        self.sigmay = [system.sigmay.copy()]
    
        self.sigmabar = [system.sigmabar]
        self.epspbar = [np.mean(system.epsp)]
        
    def add_observation(self, system):
        #TODO: check if using an array (knowing nsteps from the beginning) makes things much faster
        self.sigmabar.append(system.sigmabar)
        self.epspbar.append(np.mean(system.epsp))
        self.sigmay.append(system.sigmay.copy())
        self.sigma.append(system.sigma.copy())
        self.epsp.append(system.epsp.copy())
    
    def process_basic(self):
        self.epspbar = np.array(self.epspbar)
        self.sigmabar = np.array(self.sigmabar)
    
    def process_extra(self):
        self.add_event_maps()
    
    def add_event_maps(self):
        self.event_maps = [(self.epsp[i]- self.epsp[i-1])!=0 
                           for i in range(1, len(self.epsp))]
        self.event_maps.insert(0, np.zeros_like(self.epsp[0])) #no event in the 0th step
        
        
    def process_stability(self, mask=None):
        
        print("Processing stability...")
        
        sigma = self.sigma
        sigmay = self.sigmay
        sigmay_mean = self._sigmay_mean
        
        #TODO: add mask to where it is infinite
        max_non_inf = np.max(sigmay_mean[np.isfinite(sigmay_mean)])
        self.stability_bins_edges = np.linspace(0,2*max_non_inf, 20)
        
        #prepare outputs
        self.stability_hist = []
        self.stability_kde = []
        
        for index in tqdm(range(len(sigma))):
            #x = stability
            if mask == None:
                x = sigmay[index] - sigma[index]
            else:
                x = sigmay[index][mask.astype('bool')] - sigma[index][mask.astype('bool')]

            n, _ = np.histogram(x, self.stability_bins_edges, density = True)
            kde_y, kde_x = pdf_kde(x.ravel()[x.ravel() < float('inf')], numPoints=257)
            self.stability_hist.append(n)
            self.stability_kde.append((kde_x.astype('float64'), kde_y.astype('float64')))
            
    def process_statistics(self, sample_start, n_bins=30, cut_at=1):
        #TODO: add condition to start sampling instead of parameter. Same for cut_at
        
        delta_sigmabar = np.diff(self.sigmabar)[sample_start:]
        unloadings = -1 * delta_sigmabar[delta_sigmabar<0]

        bin_edges = np.logspace(np.log10(np.min(unloadings)),np.log10(np.max(unloadings)), n_bins)
        
        self.statistics_hist = np.hist(unloadings, bins=bin_edges, density=True)
        
        #Fitting
        centers = np.sqrt(bin_edges[0:-1] * bin_edges[1:]) #use geometric means for centers
        #set range
        if type(cut_at) == list:
            centers = centers[cut_at[0]:cut_at[1]]
            values = self.statistics_hist[0][cut_at[0]:cut_at[1]]
        else:
            last_index = int(cut_at*centers.size)
            centers = centers[0:last_index]
            values = self.statistics_hist[0][0:last_index]

        #Do the power law fit:
        _, a = power_law_fit(centers, values)
        self.exponent = a
        #TODO: for analysis, make sure to verify that the fits are okay, and maybe correct them by hand