#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from plot_func import *
from generate_fields import *
from get_corr_function import *

#%% Simulation class
class Simulation(object):
    def __init__(self, beta, s, K):
        self.beta = beta
        self.s = s
        self.K = K


#%% Generate fields
N = 1000
xi = 1000
beta_list = np.linspace(0.5,2,16)

np.random.seed(123)
u, u_t = generate_u(N)

def process(beta):
    C, C_t = generate_C(N, xi, beta)

    #s as a convolution
    s_t = C_t * u_t
    s = fft.ifft2(s_t)

    #get correlation function
    K_map, K = get_corr_function(s_t)

    #put everything in simulation
    simulation = Simulation(beta,s,K)
    return simulation

simulations = list()
for beta in beta_list:
    simulations.append(process(beta))


# %% Show s and correlations (function)
def show_s_corr(sim):
    plt.figure(figsize = (20,8), dpi = 80)
    plt.suptitle(f'N = {N}, xi = {xi}, beta = {sim.beta}', fontsize = 30)
    
    plt.subplot(1,2,1)
    plot_map(sim.s.real,'s')
    
    plt.subplot(1,2,2)
    plt.loglog(np.arange(sim.K.size//2), np.flip(sim.K.real[0:sim.K.size//2]))

# %% Show all betas
for sim in simulations:
    show_s_corr(sim)

# %%
