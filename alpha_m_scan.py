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


#%% Generate scan of simulations ------------------------------------------------------
L_list = np.linspace(100,3000, 10)
xi = float('inf')
beta_list = np.linspace(0.05,1,15)
#%%
sims = scan(L_list = L_list, xi_list = [xi], beta_list = beta_list, \
     vary_seed = True)

#%% Extract values we want
df = get_values(sims, get_alpha = True)

#%% Write df to csv
df.to_csv('alpha_simulations')
#%% Plot measured alpha vs predicted ones
alpha_predicted = 2 * (1 - beta_list)

df = pd.read_csv('alpha_simulations')

plt.figure()
plt.plot(beta_list, alpha_predicted, color = 'r', label = 'predicted')

for L in L_list[[0,3,6,9]]:
     plt.plot(df[df.L == int(L)]["beta"], df[df.L == int(L)]["alpha"], label = f'L = {int(L)}', marker = 'o')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\alpha$')
plt.legend()

plt.savefig('results/alpha_m_scan.png')

# %%
