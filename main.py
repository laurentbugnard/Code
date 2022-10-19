#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from plot_func import *
from get_corr_function import *
from Simulation import *


#%% Betascan - generate simulations
N = 1000
xi = 10000
beta_list = np.linspace(0.5,2,16)

simulations = list()
for beta in beta_list:
    sim = Simulation(N, xi , beta)
    sim.generate_fields()
    simulations.append(sim)

# %% Betascan - show
for sim in simulations:
    sim.show_s()