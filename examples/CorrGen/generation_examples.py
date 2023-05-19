#%% Imports
import os
os.chdir('C:/Users/laure/Documents/EPFL/MA/MA1/PCSL/EPM')
import sys
sys.path.append('./modules')
from config.ipy_config import ipy_config
ipy_config()
from Plotting.plot_func import *
from CorrGen.CorrGen import *


#%% Generate 2 simulations
alpha_example = CorrGen(L=100, xi=float('inf'))
alpha_example.generate_fields(method='alpha', exponent=0.4)
alpha_example.generate_sigmaY(p = 0.1)

beta_example = CorrGen(L = 100, xi = float('inf'))
beta_example.generate_fields(method='beta', exponent=0.8)
beta_example.generate_sigmaY(p = 0.1)
# %%
alpha_example.show_plots()
plt.savefig(f'examples/CorrGen/alpha_example.png')
beta_example.show_plots()
plt.savefig(f'examples/CorrGen/beta_example.png')

#%%
alpha_example.show_final()
plt.savefig(f'examples/CorrGen/alpha_example_final.png')
beta_example.show_final()
plt.savefig(f'examples/CorrGen/beta_example_final.png')
# %%