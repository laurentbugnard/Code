#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from plot_func import *
from get_corr_function import *
from Simulation import *
from scan import *


#%% Generate scan of simulations
L = 100
xi = 1e6
beta_list = np.linspace(0.1,2,10)

sims = scan('beta', beta_list, L = L, xi = xi, \
     s_center_norm = True)

#%% Show a simulation
sims[3].show_plots()
# %%