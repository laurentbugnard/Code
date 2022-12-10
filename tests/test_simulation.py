#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from plot_func import *
from get_corr_function import *
from CorrGen import *
from scan import *
from get_values import *
from power_law_fit import *
from sklearn.linear_model import LinearRegression
from config.ipy_config import*
ipy_config()

#%% Generate ONE simulation to visualize stuff
sim = CorrGen(L = 100, xi = float('inf'))
sim.generate_fields(method = 'alpha', exponent = 0.1)
sim.generate_sigmaY(p = 0.01)

sim2 = CorrGen(L = 100, xi = 5)
sim2.generate_fields(method = 'alpha', exponent = 0.1)
sim2.generate_sigmaY(p = 0.01)

# %%
sim.show_plots()
sim2.show_plots()
#%% Show s correlations

sim.corr(cut = 0.1)
sim2.corr(cut = 0.1)
# %%
sim.show_final()
sim2.show_final()

# %%
