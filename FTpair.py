#%%
import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt

def cfft(f, d):
    k_flipped = fft.fftfreq(f.shape[0]) * f.shape[0]
    k = fft.fftshift(k_flipped)
    
    if (d == 1):
        return np.exp(1j * k * np.pi) * fft.fft(f)
    elif(d == 2):
        k1 = k
        k2 = k.copy()
        k_matrix = k1 + k2.reshape(-1,1)
        return np.exp(1j* k_matrix * np.pi) * fft.fft2(f)


class FTpair(object):
    def __init__(self,f,L):
        self.f = f
        self.N = f.shape[0]
        self.L = L
        
        #real space discretization
        self.n = np.arange(self.N)
        self.deltax = self.L/self.N

        self.x = -L/2 + self.n * self.deltax

        #Fourier space discretization
        k_flipped = fft.fftfreq(self.N) * self.N
        self.k = fft.fftshift(k_flipped)
        self.deltaq = 1/self.L

        self.q = self.k * self.deltaq

        #Fourier transform (centered convention!)
        d = f.ndim
        self.f_t = fft.fftshift(cfft(self.f, d))
    
    def summary(self):
        plt.figure()
        plt.suptitle(f'N = {self.N}, L = {self.L}, dx = {self.deltax}, dq = {self.deltaq}, Nyquist_freq = {-np.min(self.q)}')
        
        plt.subplot(1,4,1)
        plt.plot(self.x, self.f, label = "f(x)")
        plt.xlabel('x')
        plt.legend()
        
        plt.subplot(1,4,2)
        plt.plot(self.q, np.abs(self.f_t), label = "abs(f_t(q))", color = 'g')
        plt.axvline(x=0, linestyle = '--', color = 'k')
        plt.xlabel('q')
        plt.legend()

        plt.subplot(1,4,3)
        plt.plot(self.q, np.real(self.f_t), label = "Re(f_t(q))", color = 'b')
        plt.axvline(x=0, linestyle = '--', color = 'k')
        plt.xlabel('q')
        plt.legend()

        plt.subplot(1,4,4)
        plt.plot(self.q, np.imag(self.f_t), label = "Im(f_t(q))", color = 'r')
        plt.axvline(x=0, linestyle = '--', color = 'k')
        plt.xlabel('q')
        plt.legend()




    

#%% Test code
L = 100
N = 100
x = np.linspace(-N/2,N/2 - 1, N)*L/N
f = np.sin(2*np.pi * 0. * x)
test = FTpair(f, L)

test.summary()
