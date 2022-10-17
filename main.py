#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from plot_func import *
from generate_fields import *
from get_corr_function import *

#%% Generate fields
N = 3000
xi = 10000
beta = 1.2

u, u_t = generate_u(N)
C, C_t = generate_C(N, xi, beta)

#s as a convolution
s_t = C_t * u_t
s = fft.ifft2(s_t)

#get correlation function
K_map, K = get_corr_function(s_t)


#%% Show results
show_plots(u, C, s, u_t, C_t, s_t, N, xi, beta)

# %% Check correlations
plt.figure()
plot_map(K_map.real, name = "corr")
plt.figure()
plt.subplot(2,1,1)
plt.plot(np.arange(K.size//2), np.flip(K.real[0:K.size//2]))
plt.subplot(2,1,2)
plt.loglog(np.arange(K.size//2), np.flip(K.real[0:K.size//2]))

# %% Show only s
plt.figure(figsize = (15,12), dpi = 80)
plt.suptitle(f'N = {N}, xi = {xi}, beta = {beta}', fontsize = 30)
plot_map(s.real,'s')

# %% Just a test nr. 2
