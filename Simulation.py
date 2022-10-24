from matplotlib import pyplot as plt
import numpy as np
from scipy import fft
from get_corr_function import get_corr_function

from plot_func import plot_map, show_plots
from power_law_fit import *


class Simulation(object):
    def __init__(self, L, xi, beta):
        self.L = L
        self.xi = xi
        self.beta = beta

    def generate_u(self, seed = 123):
        np.random.seed(seed)
        self.u = np.random.randn(self.L,self.L) #Gaussian distribution
        self.u_t = fft.fft2(self.u)
    
    def generate_C(self):
        #generate the field of q-norms
        qx,qy = np.ogrid[-self.L/2:self.L/2,-self.L/2:self.L/2]
        normes = np.sqrt(qx**2 + qy**2)

        #C_t how we would write it on paper
        C_t_centered = 1/(normes**self.beta + self.xi**(-self.beta))

        #C_t shifted (how the computer wants it)
        self.C_t = fft.ifftshift(C_t_centered)
        self.C = fft.ifft2(self.C_t)
    
    def generate_s(self, center_norm = False):
        self.s_t = self.C_t * self.u_t
        self.s = fft.ifft2(self.s_t)
        
        #if we asked it to be centered around 0
        if(center_norm):
            print("Warning: s is centered and normalized") #announce that s was modified!
            self.s = self.s - np.mean(self.s) # make it around 0
            self.s = self.s / np.std(self.s) #get a standard deviation 1
            self.s_t = fft.fft2(self.s) #redefine the FT to be consistent (and to be able to use correlations)
    
    #convenience function to generate all the fields
    def generate_fields(self, seed = 123, s_center_norm = False):
        self.generate_u(seed)
        self.generate_C()
        self.generate_s(s_center_norm)
    
    #show the s-field and also a regression for its correlations
    def show_s(self):
        plt.figure(figsize = (20,8), dpi = 80)
        plt.suptitle(f'L = {self.L}, xi = {self.xi}, beta = {self.beta}', fontsize = 30)
        
        plt.subplot(1,3,1)
        plot_map(self.s.real,'s')

        #Plot the regression
        plt.subplot(1,3,2)
        c, a = self.regression()
        K = self.get_s_corr() #get the correlations explicitely
        x = np.arange(1,K.size+1) #shifted to avoid division by 0
        y = power_law(x,c,a)
        plt.plot(x, y)

        #Plot the correlations
        plt.plot(x, K)

        #Again, but in loglog
        plt.subplot(1,3,3)
        plt.loglog(x, K)
        plt.loglog(x, y)
        plt.text(1,K[1],f'slope = {a}', bbox={'facecolor': 'white', 'pad': 3})
        
        plt.show()

    #Plot u and its correlations (to verify it is not correlated)
    def show_u(self):
        plt.figure(figsize = (20,8), dpi = 80)
        plt.suptitle(f'L = {self.L}', fontsize = 30)
        
        #u
        plt.subplot(1,2,1)
        plot_map(self.u,'u')

        #u correlations
        plt.subplot(1,2,2)
        K = self.get_u_corr()
        plt.plot(np.arange(K.size//2), np.flip(K.real[0:K.size//2]))
        
        plt.show()

    #Wrap functions to get the correlations of s and u using their FT
    def get_s_corr(self, full_map = False):
        return get_corr_function(self.s_t, full_map)
    
    def get_u_corr(self):
        return get_corr_function(self.u_t)

    #Do a power-law regression on the s-correlations
    def regression(self):
        K = self.get_s_corr()
        x = np.arange(1,K.size+1)
        c, a = power_law_fit(x,K)
        return c, a
    
    #Calling the show_plots from the other file
    def show_plots(self):
        show_plots(self.u, self.C, self.s, \
            self.u_t, self.C_t, self.s_t, \
                self.L, self.xi, self.beta)