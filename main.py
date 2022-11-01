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
test_sim = Simulation(L = 9, xi = 1e6, beta = 0.8)
test_sim.generate_fields(s_centered = True)

#%% Show all plots
test_sim.show_plots()

#%% Show s correlations
test_sim.show_s






#%% Generate scan of simulations ------------------------------------------------------
L_list = np.linspace(100,1000,10)
xi = 1e6
beta_list = np.linspace(0.05,1,100)

sims = scan(L_list = L_list, xi_list = [xi], beta_list = beta_list, \
     s_centered = True, vary_seed = True)

#%% Extract values we want
df = get_values(sims)

#%% Plot L vs. beta for different beta
plt.figure()

slopes = []
for beta in beta_list:
     filtered_df = df[df.beta == beta]
     slope = plot_power_law_fit(filtered_df.L, filtered_df.std_s, label = f'beta = {beta}')
     slopes.append(slope)

#%% Plot slope vs beta
plt.figure()
plt.scatter(beta_list,slopes)
plt.xlabel("beta")
plt.ylabel("slope of std_s(L)")

#%% Perform linear regression on it and replot
reg = LinearRegression().fit(beta_list.reshape(-1,1),slopes)
plt.figure()
plt.scatter(beta_list, slopes, label = "data")
plt.plot(beta_list, reg.coef_ * beta_list + reg.intercept_, \
     label = f'{reg.coef_[0]:.2f}*beta + {reg.intercept_:.2f}', color = 'k')
plt.plot(beta_list, -2*(1-beta_list)+3, label = "test", color = 'r')
plt.xlabel("beta")
plt.ylabel("slope of std_s(L)")
plt.legend()

