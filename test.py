#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from plot_func import *
from get_corr_function import *
from Simulation import *
from scan import *
from get_values import *
from power_law_fit import *
from sklearn.linear_model import LinearRegression
from FTpair import *
from ipy_config import*
ipy_config()

#%%
#generate s and invert it with respecting periodic BC
s = np.linspace(-5,5,100)

s = np.exp(x)
sprime = np.concatenate([[s[0]], s[1:][::-1]])

#%%
#compute both Fourier transforms
s = FTpair(s, L = s.size)
sprime = FTpair(sprime, L = sprime.size)

#%% 
#check that the Fourier transfroms are indeed conjugates
s.summary()
sprime.summary()

print(np.allclose(s.f_t, np.conj(sprime.f_t)))









# %%2D test
u = np.random.randn(10,10)
#inverser de manière à respecter les BC périodiques
u_interm = np.concatenate([u[:,0].reshape(-1,1) , u[:,1:][:,::-1]], axis = 1)
u_prime = np.concatenate([[u_interm[0]], u_interm[1:][::-1]], axis = 0)

#%% FT classique
u_fft = fft.fftshift(fft.fft2(u))
u_prime_fft = fft.fftshift(fft.fft2(u_prime))

#%% FT avec FTpair
u = FTpair(u, L = u.shape[0])
u_prime = FTpair(u_prime, L = u_prime.shape[0])

#%%
#FONCTIONNE: on voit bien que dans tous les cas la valeur absolue de FT 
#est pareille, et donc on peut choisir n'importe quelle convention si on 
#s'intéresse pas à Re ou Im.

plt.figure()
plt.suptitle("u et sa transformée (scipy)")
plt.subplot(1,4,1)
plt.imshow(u.f)
plt.subplot(1,4,2)
plt.title("||")
plt.imshow(np.abs(u_fft))
plt.subplot(1,4,3)
plt.title("Re")
plt.imshow(np.real(u_fft))
plt.subplot(1,4,4)
plt.title("Im")
plt.imshow(np.imag(u_fft))


plt.figure()
plt.suptitle("u_prime et sa transformée (scipy)")
plt.subplot(1,4,1)
plt.imshow(u_prime.f)
plt.subplot(1,4,2)
plt.title("||")
plt.imshow(np.abs(u_prime_fft))
plt.subplot(1,4,3)
plt.title("Re")
plt.imshow(np.real(u_prime_fft))
plt.subplot(1,4,4)
plt.title("Im")
plt.imshow(np.imag(u_prime_fft))

plt.figure()
plt.suptitle("u_prime et sa transformée (FTpair)")
plt.subplot(1,4,1)
plt.imshow(u_prime.f)
plt.subplot(1,4,2)
plt.title("||")
plt.imshow(np.abs(u_prime.f_t))
plt.subplot(1,4,3)
plt.title("Re")
plt.imshow(np.real(u_prime.f_t))
plt.subplot(1,4,4)
plt.title("Im")
plt.imshow(np.imag(u_prime.f_t))

# %%
print(np.allclose(u.f_t, np.conj(u_prime.f_t)))

#%% Test fonction de corrélation
plt.figure()
plt.imshow(np.abs(fft.ifftshift(fft.ifft2(np.abs(u.f_t*u_prime.f_t)))))
plt.figure()
plt.imshow(np.abs(get_corr_function(u.f, full_map = True)))
# %%
