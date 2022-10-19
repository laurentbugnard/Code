from matplotlib import pyplot as plt
import numpy as np
from scipy import fft
from get_corr_function import get_corr_function

from plot_func import plot_map


class Simulation(object):
    def __init__(self, N, xi, beta):
        self.N = N
        self.xi = xi
        self.beta = beta

    def generate_u(self, seed = 123):
        np.random.seed(seed)
        self.u = np.random.randn(self.N,self.N)
        self.u_t = fft.fft2(self.u)
    
    def generate_C(self):
        qx,qy = np.ogrid[-self.N/2:self.N/2,-self.N/2:self.N/2]
        normes = np.sqrt(qx**2 + qy**2)

        C_t_centered = 1/(normes**self.beta + self.xi**(-self.beta))

        self.C_t = fft.ifftshift(C_t_centered)
        self.C = fft.ifft2(self.C_t)
    
    def generate_s(self):
        self.s_t = self.C_t * self.u_t
        self.s = fft.ifft2(self.s_t)
    
    def generate_fields(self, seed = 123):
        self.generate_u(seed)
        self.generate_C()
        self.generate_s()
    
    def show_s(self):
        plt.figure(figsize = (20,8), dpi = 80)
        plt.suptitle(f'N = {self.N}, xi = {self.xi}, beta = {self.beta}', fontsize = 30)
        
        plt.subplot(1,2,1)
        plot_map(self.s.real,'s')

        plt.subplot(1,2,2)
        K = self.get_s_corr()
        plt.loglog(np.arange(K.size//2), np.flip(K.real[0:K.size//2]))
        
        plt.show()

    def show_u(self):
        plt.figure(figsize = (20,8), dpi = 80)
        plt.suptitle(f'N = {self.N}', fontsize = 30)
        
        plt.subplot(1,2,1)
        plot_map(self.u,'u')

        plt.subplot(1,2,2)
        K = self.get_u_corr()
        plt.plot(np.arange(K.size//2), np.flip(K.real[0:K.size//2]))
        
        plt.show()

    def get_s_corr(self):
        return get_corr_function(self.s_t)
    
    def get_u_corr(self):
        return get_corr_function(self.u_t)