from matplotlib import pyplot as plt
import numpy as np
from scipy import fft
from get_corr_function import get_corr_function

from plot_func import plot_map, plot_ft
from power_law_fit import *

#TODO supprimer le correlation factor


class Simulation(object):
    def __init__(self, L, xi):
        self.L = int(L)
        self.xi = xi

    def generate_u(self, seed = 123):
        np.random.seed(seed)
        self.u = np.random.randn(self.L,self.L) #Gaussian distribution
        self.u_t = fft.fft2(self.u)
        self.seed = seed
    
    def generate_C(self, method, exponent, s_centered = True):
        if(method == 'beta'):
            self.method = 'beta'
            self.beta = exponent
            #generate the field of q-norms
            qx,qy = np.meshgrid(fft.fftfreq(self.L), fft.fftfreq(self.L))
            normes = np.sqrt(qx**2 + qy**2)

            #C_t how we would write it on paper
            self.C_t = 1/(normes**self.beta + self.xi**(-self.beta))
        
        elif(method == 'alpha'):
            self.method = 'alpha'
            self.alpha = exponent
            x, y = np.meshgrid(fft.fftfreq(self.L)*self.L,fft.fftfreq(self.L)*self.L)
            normes = np.sqrt(x**2 + y**2)
            g = reg_power_law(normes, 1, self.alpha) * np.exp(-normes/self.xi)
            g_t = fft.fft2(g).real
            self.C_t = np.sqrt(g_t+0j)
            #TODO tenter de mettre l'exponential cutoff directement dans le corrélateur, pour voir si ça fait une différence
        #if we asked it to be centered around 0
        if(not s_centered):
            print("Warning: <s> is not 0!") #announce that s was modified!
        else:
            self.C_t[0,0] = 0
        
        self.C = fft.ifft2(self.C_t)
    
    def generate_s(self, s_normalized = False):
        self.s_t = self.C_t * self.u_t
        self.s = fft.ifft2(self.s_t)
        if(s_normalized): self.s = self.s / np.std(self.s)
    
    def generate_sigmaY(self, p = 1):
        self.p = p
        self.sigmaY = np.exp(p * np.real(self.s))
    
    #convenience function to generate all the fields
    def generate_fields(self, method, exponent, seed = 123, s_centered = True, s_normalized = True):
        self.generate_u(int(seed))
        self.generate_C(method = method, exponent = exponent, s_centered = s_centered)
        self.generate_s(s_normalized)
    
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