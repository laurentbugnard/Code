#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from plot_func import *
from get_corr_function import *
from Simulation import *


#%% Betascan - generate simulations
L = 100
xi = 1e6
beta_list = np.linspace(0.1,2,10)

simulations = list()
for beta in beta_list:
    sim = Simulation(N, xi , beta)
    sim.generate_fields(seed = 1)
    simulations.append(sim)

# %% Betascan - show
for sim in simulations:
    sim.show_s()
# %% Show only one simulation
simulations[3].show_plots()

<<<<<<< HEAD
# %% Just a test nr. 2
=======
# %% Regression test
y = simulations[5].get_s_corr()
x = np.arange(1,y.size+1)
#y_inverted = 1/y
c, a = power_law_fit(x,y)
y2 = power_law(x, c, a)
#y2_inverted = power_law(x, c, a)
plt.plot(x,y)
plt.plot(x,y2)
#plt.plot(x,y2_inverted)
# %%
>>>>>>> betascan
