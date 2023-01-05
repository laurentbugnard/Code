from matplotlib import pyplot as plt
import numpy as np
from scipy import fft
import pandas as pd
import sys

from plot_func.plot_func import plot_map, plot_ft, power_law, power_law_fit, regularized_power_law

class CorrGen(object):
    """Object used to generate a random but power-law-correlated yield stress field, like described in the README.
    First, create the object using the length ``L`` and the cutoff size ``xi``. Then, generate power correlations
    using ``generate_fields`` and the desired parameters ``method`` and ``exponent``. Finally, generate the yield stress field
    with ``generate_sigmay`` and the parameter ``p``.

    Attributes:
         L (int): Linear dimension in pixels of the (square) system.
         xi (float): Cutoff size for the correlation function.
         
         seed (int): Seed to generate the gaussian field ``u``.
         u (np.ndarray): Uncorrelated gaussian field with mean 0 and std 1.
         u_t (np.ndarray): Fourier transform of ``u``
         
         method (str): Method used to find the correlator ``C``. Either "alpha" or "beta".
         alpha (float): Exponent of the correlation function in the ``method = "alpha"`` case.
         beta (float): Exponent of the correlator in the ``method = "beta"`` case.
         C (np.ndarray): The correlator.
         C_t (np.ndarray): Fourier transform of ``C``.
         
         s (np.ndarray): Random but power-law-correlated field.
         s_t (np.ndarray): Fourier transform of ``s``.
         
         p (float): Parameter to describe the relationship between ``s`` and ``sigmaY``.
         sigmaY (np.ndarray): Final result, yield stress field.

    """
    L:int
    xi:float
    
    seed:int
    u:np.ndarray
    u_t:np.ndarray
    
    method:str
    alpha:float
    beta:float
    C:np.ndarray
    C_t:np.ndarray
    
    s:np.ndarray
    s_t:np.ndarray
    
    p:float
    sigmaY:np.ndarray
    
    def __init__(self, L:int, xi:float):
        """Initialize the generation object with most general parameters.

        Args:
            L (int): Linear dimension in pixels of the (square) system.
            xi (float): Cutoff size for the correlation function.
        """
        self.L = int(L)
        self.xi = xi
    

    def generate_fields(self, method:str, exponent:float, seed = 123, s_centered = True, s_normalized = True, reg_value = 0.0):
        """Generate the final field ``s`` and the intermediary ``u`` and ``C`` fields and their Fourier transforms.
        The ``method`` parameter to compute ``C`` can be either "alpha" or "beta", for numerical or analytical ``C`` respectively.

        Args:
            method (str): "alpha" for numerical method, "beta" for analytical method.
            exponent (float): alpha or beta exponent, depending on the method chosen.
            seed (int, optional): seed to generate the gaussian field ``u``. Defaults to 123.
            s_centered (bool, optional): determines whether ``s`` should have mean = 0 (imposed through ``mean(u)`` = 0). Defaults to True.
            s_normalized (bool, optional): determines whether ``s`` should have std = 1. Defaults to True.
            reg_value (float, optional): the value we should use for the regularization of the power law function
            (for the correlation function in the alpha method).
        """
        
        
        ########GENERATE U##########
        self.seed = seed
        np.random.seed(seed)
        self.u = np.random.randn(self.L,self.L) #Uncorrelated gaussian map

        if(not(s_centered)):
            print('Warning: <s> =/= 0')
        else:
            self.u -= np.mean(self.u) #shift mean to have both u and s centered around 0
            
        self.u_t = fft.fft2(self.u)
        
        ########GENERATE C##########
        if(method == 'alpha'):
            self.method = 'alpha'
            self.alpha = exponent
            self.beta = None
            
            #generate meshgrid to compute power law (gamma)
            x, y = np.meshgrid(fft.fftfreq(self.L)*self.L,fft.fftfreq(self.L)*self.L)
            normes = np.sqrt(x**2 + y**2)

            gamma = regularized_power_law(normes, 1, self.alpha, reg_value=reg_value) * np.exp(-normes/self.xi)
            
            gamma_t = fft.fft2(gamma).real #.real just to avoid imaginary noise
            self.C_t = np.sqrt(gamma_t+0j) #TODO check why imaginary: because of negative values in the root?
            #TODO tenter de mettre l'exponential cutoff directement dans le corrélateur, pour voir si ça fait une différence
        
        elif(method == 'beta'):
            self.method = 'beta'
            self.beta = exponent
            self.alpha = None
            
            #generate the field of q-norms
            qx,qy = np.meshgrid(fft.fftfreq(self.L), fft.fftfreq(self.L))
            normes = np.sqrt(qx**2 + qy**2)

            self.C_t = 1/(normes**self.beta + self.xi**(-self.beta))
            self.C_t[0,0] = 0 #REGULARIZE to avoid errors. 
            #TODO: find out which is the best regularization
        
        self.C = fft.ifft2(self.C_t) #add C for completeness
        
        #########GENERATE S############
        self.s_t = self.C_t * self.u_t
        self.s = fft.ifft2(self.s_t)
        if(s_normalized): self.s = self.s / np.std(self.s)
    
    
    def generate_sigmaY(self, p = 1):
        """Generate the yield stress field (final result) from the power-law-correlated field ``s``,
        using :math:``\sigma^Y = \exp(ps)``.

        Args:
            p (int, optional): The scaling in the exponential. Defaults to 1.
        """
        self.p = p
        self.sigmaY = np.exp(p * np.real(self.s))
    
    

    def measure_corr(self, fourier=False, mean_window = 0, cut = 1, plot = True, scale='log') -> float:
        """Measures the correlation functions for ``u``, ``C`` and ``s`` and plots them, along with a
        power law fit in the ``s`` case. Returns the exponent of that fit.

        Args:
            fourier (bool, optional): Determines whether the Fourier-space correlation function should be used. Defaults to False.
            mean_window (int, optional): Window to perform the moving mean before fitting s correlations. Defaults to 0.
            cut (int, optional): Fraction of data considered in the regression. Defaults to 1.
            plot (bool, optional): Determines whether to show the plots or just compute the fitted exponent. Defaults to True.
            scale (str, optional): The scale of the plots: either ``'linear'``,``'log'`` or ``'semilog'``. Defaults to 'log'.

        Returns:
            float: Exponent of the power law fit.
        """

        #get the correlation functions
        K_u = get_corr_function(self.u, fourier=fourier)
        K_C = get_corr_function(self.C, fourier=fourier)
        K_s = get_corr_function(self.s, fourier=fourier)
        
        #prepare the x-axis
        x = np.arange(K_s.size)
        #cut off the part not wanted for the fit
        cut_begin_C = 2 #cut out the correlation function at 0 (no meaning)
        cut_end_C = x.size
        cut_begin_s = 2 
        cut_end_s = int(x.size*cut)
            
        #### FITTING s correlation function
        def fit_corr(K, cut_begin, cut_end):
            #Smoothen
            K_smooth = K
            if (mean_window != 0):
                K_smooth = np.convolve(K, np.ones(mean_window)/mean_window, mode='same') #moving mean
            
            K_cut = K_smooth[cut_begin:cut_end]
            x_cut = x[cut_begin:cut_end]

            #fit and predict (for the plot)
            c, a = power_law_fit(x_cut,K_cut)
            y = power_law(x,c,a)
            
            return y, a
        
        y_C, a_C = fit_corr(K_C, cut_begin_C, cut_end_C)
        y_s, a_s = fit_corr(K_s, cut_begin_s, cut_end_s)
        

        if(plot):
            #prepare figure
            plt.figure(figsize = (20,8), dpi = 80)
            if self.method == 'alpha':
                plt.suptitle(fr'$L = {self.L}, \xi = {self.xi}, \alpha = {self.alpha}$', fontsize = 30)
            elif self.method == 'beta':
                plt.suptitle(fr'$L = {self.L}, \xi = {self.xi}, \beta = {self.beta}$', fontsize = 30)
            
            #plot u
            plt.subplot(2,3,1)
            plot_map(self.u,r'$u$')
            #plot C
            plt.subplot(2,3,2)
            plot_map(self.C.real,r'$Re(C)$', centered=True)
            #plot s
            plt.subplot(2,3,3)
            plot_map(self.s.real,r'$Re(s)$')

            
            #Plot the correlations and regression
            #u
            plt.subplot(2,3,4)
            if fourier: plt.title(r'$\tilde\Gamma_u$')
            else: plt.title(r'$\Gamma_u$')
            plt.plot(x, K_u)
            if scale == 'log': plt.xscale('log'); plt.yscale('log')
            elif scale == 'semilog': plt.yscale('log')
            plt.xlabel(r'$r$')
            if fourier: plt.ylabel(r'$\tilde\Gamma (r)$')
            else: plt.ylabel(r'$\Gamma (r)$')
            #C
            plt.subplot(2,3,5)
            if fourier: plt.title(r'$\tilde\Gamma_C$')
            else: plt.title(r'$\Gamma_C$')
            plt.plot(x, K_C)
            if scale == 'log': plt.xscale('log'); plt.yscale('log')
            elif scale == 'semilog': plt.yscale('log')
            plt.plot(x, y_C, color = 'r', label = fr'power law fit, $\alpha_m = {a_C:.4f}$')
            plt.axvline(x = x[cut_begin_C], linestyle = '--', color = 'k')
            plt.axvline(x = x[cut_end_C-1], linestyle = '--', color = 'k')
            plt.xlabel(r'$r$')
            plt.legend()
            #s
            plt.subplot(2,3,6)
            if fourier: plt.title(r'$\tilde\Gamma_s$')
            else: plt.title(r'$\Gamma_s$')
            plt.plot(x, K_s)
            if scale == 'log': plt.xscale('log'); plt.yscale('log')
            elif scale == 'semilog': plt.yscale('log')
            plt.plot(x, y_s, color = 'r', label = fr'power law fit, $\alpha_m = {a_s:.4f}$')
            plt.axvline(x = x[cut_begin_s], linestyle = '--', color = 'k')
            plt.axvline(x = x[cut_end_s-1], linestyle = '--', color = 'k')
            plt.xlabel(r'$r$')
            plt.legend()
            
  
  
            plt.gcf().tight_layout()
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()

        return a_s

    def show_u(self):
        """This function just shows ``u`` and its correlation function, which should 0 everywhere 
        (with fluctuations due to finite system size).
        """
        plt.figure(figsize = (20,8), dpi = 80)
        plt.suptitle(f'L = {self.L}', fontsize = 30)
        
        #u
        plt.subplot(1,2,1)
        plot_map(self.u,'u')

        #u correlations
        plt.subplot(1,2,2)
        K = get_corr_function(self.u)
        plt.plot(np.arange(K.size//2), np.flip(K.real[0:K.size//2]))
        
        plt.show()
    
    def show_plots(self):
        """Visualization function, which shows all of the fields used in the procedure to generate ``s`` 
        (``u``,``u_t``,``C``,``C_t``,``s``,``s_t``).
        """
        plt.figure(figsize = (25,12), dpi = 80)
        
        if(self.method == 'beta'):
            plt.suptitle(fr'$L = {self.L}, \xi = {self.xi}, \beta = {self.beta}$', fontsize = 30)
        else:
            plt.suptitle(fr'$L = {self.L}, \xi = {self.xi}, \alpha = {self.alpha}$', fontsize = 30)
        
        #Maps
        plt.subplot(2,3,1)
        plot_map(self.u,r'$u$')
        
        plt.subplot(2,3,2)
        plot_map(self.C.real,r'$Re(C)$', centered = True)
        
        plt.subplot(2,3,3)
        plot_map(self.s.real,r'$Re(s)$')
        
        
        #Their FT
        plt.subplot(2,3,4)
        plot_ft(self.u_t,r'$|\tilde{u}|$')
        
        plt.subplot(2,3,5)
        plot_ft(self.C_t,r'$|\tilde{C}|$', logscale = True)
        
        plt.subplot(2,3,6)
        plot_ft(self.s_t,r'$|\tilde{s}|$', logscale = True)

        #tight and maximize window
        plt.gcf().tight_layout()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()
    
    def show_final(self):
        """Visualization function which shows ``s`` and ``sigmay``.
        """
        
        plt.figure(figsize = (20,8), dpi = 80)
        if(self.method == 'beta'):
            plt.suptitle(fr'$L = {self.L}, \xi = {self.xi}, \beta = {self.beta}$', fontsize = 30)
        else:
            plt.suptitle(fr'$L = {self.L}, \xi = {self.xi}, \alpha = {self.alpha}$', fontsize = 30)
        
        #s
        plt.subplot(1,2,1)
        plot_map(self.s.real,r'$Re(s)$')

        # #histogram of s values (to see its range)
        # plt.subplot(1,3,2)
        # plt.title('Histogram of s values')
        # plt.hist(np.ravel(self.s.real), bins = 50)

        #sigmaY
        sp = plt.subplot(1,2,2)
        plot_map(self.sigmaY, fr'$\sigma^Y, p = {self.p}$')
        
        #configure colorbar
        middle = np.mean(self.sigmaY)
        min = middle - np.std(self.sigmaY)
        max = middle + np.std(self.sigmaY)
        plt.clim(min, max)
        cbar = plt.gca().images[-1].colorbar
        cbar.set_ticks([min,middle,max])
        cbar.set_ticklabels([f'(mean - std) = {min:.2f}',f'mean = {middle:.2f}',f'(mean + std) = {max:.2f}'])
        plt.show()

    def get_coordinates(self) -> np.ndarray:
        """Returns the real coordinates one should use with the system.

        Returns:
            np.ndarray: Meshgrid of coordinates.
        """
        interval = np.arange(self.L) - int(self.L//2)
        return np.meshgrid(interval, interval)
    


#################FUNCTIONS#######################
def scan(L_list:list[int], xi_list:list[float], exponent_list:list[float], 
         method = 'alpha', s_centered = True, s_normalized = False, vary_seed = False) -> list[CorrGen]:
    """Helper function to generate a list of CorrGen objects, with parameters ``L``, ``xi`` and ``exponent`` corresponding to 
    all possible combinations of parameters given in ``L_list``, ``xi_list`` and ``exponent_list``. It is possible to vary the ``seed``
    at each generation using ``vary_seed = True``.

    Args:
        L_list (list[int]): List of lengths ``L``.
        xi_list (list[float]): List of cutoff sizes ``xi``.
        exponent_list (list[float]): List of ``exponent``s, either alpha or beta according to ``method``.
        method (str, optional): Method used to generate ``C``. Defaults to 'alpha'.
        s_centered (bool, optional): Determines whether ``s`` should have mean = 0 (imposed through ``C_t(0)`` = 0). Defaults to True.
        s_normalized (bool, optional): Determines whether ``s`` should have std = 1. Defaults to False.
        vary_seed (bool, optional): Determines whether the ``seed`` should be changed for each simulation to avoid correlations
        between realisations of the system. Defaults to False.

    Returns:
        list[CorrGen]: List of CorrGen objects.
    """

    # Prepare list of seeds
    n = len(L_list) * len(xi_list) * len(exponent_list) #total number of simulations
    if(vary_seed):
        seed_list = np.round(np.random.uniform(0,1000, n)) #pick n random seeds
    else:
        seed_list = np.ones(n)
        
        
    i = 0 #initialize iterator for seeds
    list_corrgen = list()
    for L in L_list:
        for xi in xi_list:
            for exponent in exponent_list:
                sim = CorrGen(L, xi)
                sim.generate_fields(method = method, exponent = exponent, s_centered = s_centered, s_normalized = s_normalized,
                                    seed = seed_list[i])
                list_corrgen.append(sim)
                i = i+1 #update iterator for seeds
                
    return list_corrgen

def get_corr_function(f:np.ndarray, fourier=False, full_map = False, normalized = True) -> np.ndarray:
    """Returns the (auto)correlation function associated with the given field ``f``. If ``full_map = True``, a 2D complex map is returned.
    Otherwise, isotropy is imposed by calling ``isotropise`` and a 1D section of the result is returned.

    Args:
        f (np.ndarray): Field for which we want the correlation function.
        fourier (bool, optional): Determines whether the Fourier-space correlation function should be returned. Defaults to False.
        full_map (bool, optional): Determines whether a full 2D complex map should be returned. Defaults to False.
        normalized (bool, optional): Determines whether the correlation function is normalized with the mean square of ``f``. Defaults to True.

    Returns:
        np.ndarray: _description_
    """
    
    #first, make sure f is real, because parasite complex values can heavily impact results
    f = f.real #TODO check if it is taking the real part that messes up correlations

    f_t = fft.fft2(f) #get the fourier transform
    
    #get correlation map (2D). See README for method used 
    if fourier: #correlations in Fourier space
        K_map = np.abs(f_t)**2/f.size 
    else:
        K_map = fft.ifft2(np.abs(f_t)**2)/f.size
    
    #isotropise to diminish fluctuations
    K_map = isotropise(K_map)
    
    #get only a section of it (1D). #TODO find better solution
    K = K_map[0,:K_map.shape[1]//2]

    
    if(full_map):
        return K_map
    else:
        if normalized & (not fourier):
            return K.real / np.var(f) #TODO change it to actual mean square (?)
        else:
            return K.real
        
def get_values(corrgen_list, get_alpha_empirical = False) -> pd.DataFrame:
    """Returns a Pandas dataframe resuming the parameters of a list of CorrGen objects. If ``get_alpha_empirical = True``,
    the correlation function is fitted to obtain the empirical exponent ``alpha`` for each simulation. 

    Args:
        corrgen_list (_type_): List of CorrGen objects.
        get_alpha_empirical (bool, optional): Determines whether the empirical exponent ``alpha`` should be computed
        using the correlation function. Defaults to False.

    Returns:
        pd.DataFrame: Pandas dataframe containing the parameters for all CorrGen objects.
    """
    
    df = pd.DataFrame(index = np.arange(len(corrgen_list)), \
        columns = ["L", "xi", "alpha", "beta", "std_s", "seed", "alpha_empirical"])
    
    for i, sim in enumerate(corrgen_list):
        df.iloc[i]["L"] = sim.L
        df.iloc[i]["xi"] = sim.xi
        df.iloc[i]["alpha"] = sim.alpha
        df.iloc[i]["beta"] = sim.beta
        df.iloc[i]["std_s"] = np.std(sim.s)
        df.iloc[i]["seed"] = sim.seed
        if(get_alpha_empirical): 
            df.iloc[i]["alpha_empirical"] = sim.corr(plot = False)
    
    return df


def isotropise(f:np.ndarray) -> np.ndarray:
    """Takes a 2D square array and returns the mean over its 4 possible 90 degrees rotations.

    Args:
        f (np.ndarray): 2D square array.

    Returns:
        np.ndarray: "Isotropized" (over 4 directions) array.
    """
    #extend periodically
    f = np.r_[f, [f[0,:]]]
    f = np.c_[f, f[:,0]]
    
    #average over 4 directions
    f_iso = 0.25*(f + np.rot90(f,1) + np.rot90(f,2) + np.rot90(f,3))
    
    return f_iso