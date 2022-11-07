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

from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("matplotlib", "qt")
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

#%% Generate ONE simulation to visualize stuff
test_sim = Simulation(L = 1000, xi = 1e8, beta = 0.6)
test_sim.generate_fields(s_centered = True, s_normalized= True)

#%% Show s correlations
test_sim.corr(cut = 0.1)
# %%
test_sim.show_plots()
# %%
