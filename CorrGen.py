from matplotlib import pyplot as plt
import numpy as np
from scipy import fft
import pandas as pd

from plot_func import plot_map, plot_ft, power_law, power_law_fit, reg_power_law

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
    
    def generate_fields(self, method:str, exponent:float, seed = 123, s_centered = True, s_normalized = True):
        """Generate the final field ``s`` and the intermediary ``u`` and ``C`` fields and their Fourier transforms.
        The ``method`` parameter to compute ``C`` can be either "alpha" or "beta", for numerical or analytical ``C`` respectively.

        Args:
            method (str): "alpha" for numerical method, "beta" for analytical method.
            exponent (float): alpha or beta exponent, depending on the method chosen.
            seed (int, optional): seed to generate the gaussian field ``u``. Defaults to 123.
            s_centered (bool, optional): determines whether ``s`` should have mean = 0 (imposed through ``C_t(0)`` = 0). Defaults to True.
            s_normalized (bool, optional): determines whether ``s`` should have std = 1. Defaults to True.
        """
        
        
        ########GENERATE U##########
        self.seed = seed
        np.random.seed(seed)
        self.u = np.random.randn(self.L,self.L) #Uncorrelated gaussian map
        self.u_t = fft.fft2(self.u)
        
        ########GENERATE C##########
        if(method == 'alpha'):
            self.method = 'alpha'
            self.alpha = exponent
            self.beta = None
            
            #generate meshgrid to compute power law (gamma)
            x, y = np.meshgrid(fft.fftfreq(self.L)*self.L,fft.fftfreq(self.L)*self.L)
            normes = np.sqrt(x**2 + y**2)
            gamma = reg_power_law(normes, 1, self.alpha) * np.exp(-normes/self.xi)
            
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
        
        if(not s_centered):
            print("Warning: <s> is not 0!")
        else:
            self.C_t[0,0] = 0
        
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
    
    
    #show the s-field and also a regression for its correlations
    def corr(self, mean_window = 0, cut = 1, plot = True):
        
        #get correlations and fit
        K = get_corr_function(self.s) #get the correlations
        K_smooth = K
        if (mean_window != 0):
            K_smooth = np.convolve(K, np.ones(mean_window)/mean_window, mode='same') #moving mean
        x = np.arange(1,K.size+1) #shifted to avoid division by 0
        
        #cut off the part not wanted for the fit
        cut_begin = 2
        cut_end = int(x.size*cut)
        K_cut = K_smooth[cut_begin:cut_end]
        x_cut = x[cut_begin:cut_end]

        c, a = power_law_fit(x_cut,K_cut) #fit
        y = power_law(x,c,a) #predicted

        if(plot):
            #prepare figure
            plt.figure(figsize = (20,8), dpi = 80)
            plt.suptitle(fr'$L = {self.L}, \xi = {self.xi}, \beta = {self.beta}$', fontsize = 30)
            #plot s
            plt.subplot(1,3,1)
            plot_map(self.s.real,r'$Re(s)$')

            #Plot the correlations and regression
            plt.subplot(1,3,2)
            plt.plot(x, K)
            plt.plot(x, y, color = 'r', label = 'power law fit')
            plt.xlabel(r'$r$')
            plt.ylabel(r'$\Gamma (r)$')

            plt.legend()


            #Again, but in loglog
            plt.subplot(1,3,3)
            plt.title(fr'$\alpha_m = {a:.2f}$', fontsize = 20)
            plt.loglog(x, K)
            if(mean_window != 0):
                plt.loglog(x, K_smooth, label = 'smooth')
            plt.loglog(x, y, color = 'r', label = 'power law fit')
            plt.axvline(x = x[cut_begin], linestyle = '--', color = 'k')
            plt.axvline(x = x[cut_end], linestyle = '--', color = 'k')
            plt.xlabel(r'$r$')
            plt.ylabel(r'$\Gamma (r)$')
            plt.legend()
            
            plt.show()

        return a

    #Plot u and its correlations (to verify it is not correlated)
    def show_u(self):
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
    
    #Calling the show_plots from the other file
    def show_plots(self):
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

        plt.show()
    
    def show_final(self):
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
        
        #configure colorabar
        middle = np.mean(self.sigmaY)
        min = middle - np.std(self.sigmaY)
        max = middle + np.std(self.sigmaY)
        plt.clim(min, max)
        cbar = plt.gca().images[-1].colorbar
        cbar.set_ticks([min,middle,max])
        cbar.set_ticklabels([f'(mean - std) = {min:.2f}',f'mean = {middle:.2f}',f'(mean + std) = {max:.2f}'])
        plt.show()

    def get_coordinates(self):
        interval = np.arange(self.L) - int(self.L//2)
        return np.meshgrid(interval, interval)
    


#################FUNCTIONS#######################
def scan(L_list, xi_list, exponent_list, method = 'alpha', s_centered = True, s_normalized = False, vary_seed = False):
    
    '''Returns a LIST of simulations over the chosen parameter.'''
    
    # make a list of seeds with which to simulate
    n = len(L_list) * len(xi_list) * len(exponent_list) #total number of simulations
    if(vary_seed):
        seed_list = np.round(np.random.uniform(0,1000, n)) #pick n random seeds
    else:
        seed_list = np.ones(n)
        

    i = 0 #initialize iterator for seeds
    simulations = list() #initialize output
    for L in L_list:
        for xi in xi_list:
            for exponent in exponent_list:
                Corrg
                sim = CorrGen(L, xi)
                sim.generate_fields(method = method, exponent = exponent, s_centered = s_centered, s_normalized = s_normalized, \
                    seed = seed_list[i])
                simulations.append(sim)
                i = i+1 #update iterator for seeds
                
    return simulations

def get_corr_function(f, full_map = False, normalized = True):
    f = f.real #first make sure it's real, because parasite complex values can change everything
    #full complex correlation map
    f_t = fft.fft2(f) #get the fourier transform
    #normalize by the number of points: shouldn't be necessary since ifft2 already does
    #don't know why, but it doesn't work without it:
    K_map = fft.ifft2(np.abs(f_t)**2)/f.size 
    K = K_map[0,:K_map.shape[1]//2]
    if(full_map):
        return K_map
    else:
        if(normalized):
            return K.real / np.var(f)
        else:
            return K.real
        
def get_values(simulation_list, get_alpha = False):
    df = pd.DataFrame(index = np.arange(len(simulation_list)), \
        columns = ["L", "xi", "beta", "std_s", "seed", "alpha"])
    
    for i, sim in enumerate(simulation_list):
        df.iloc[i]["L"] = sim.L
        df.iloc[i]["xi"] = sim.xi
        df.iloc[i]["beta"] = sim.beta
        df.iloc[i]["std_s"] = np.std(sim.s)
        df.iloc[i]["seed"] = sim.seed
        if(get_alpha): 
            df.iloc[i]["alpha"] = sim.corr(plot = False)
    
    return df