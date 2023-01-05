#%% Imports
import os
os.chdir('C:/Users/laure/Documents/EPFL/MA/MA1/PCSL/EPM')
import sys
sys.path.append('./modules')
from config.ipy_config import ipy_config
ipy_config()
import copy
from plot_func.plot_func import *
from CorrGen.CorrGen import *

#%%
sim = CorrGen(L = 100, xi = float('inf'))

# beta1 = copy.copy(sim)
# beta1.generate_fields(method = 'beta', exponent= 0.8)

# beta2 = copy.copy(sim)
# beta2.generate_fields(method = 'beta', exponent= 0.3)

# beta3 = copy.copy(sim)
# beta3.generate_fields(method = 'beta', exponent= 1.1)

# alpha1 = copy.copy(sim)
# alpha1.generate_fields(method = 'alpha', exponent= 0.1)

# alpha2 = copy.copy(sim)
# alpha2.generate_fields(method = 'alpha', exponent= 0.8)

alpha3 = copy.copy(sim)
alpha3.generate_fields(method = 'alpha', exponent=0.8, seed=7)

# %% Show alphas
# alpha1.corr(cut = 0.1)
# # plt.savefig(f'../images/corr_beta={sim1.beta}.png')
# alpha2.corr(cut = 0.1)
# plt.savefig(f'../images/corr_beta={sim2.beta}.png')
alpha3.measure_corr(scale='log',fourier=False, cut = 0.3)
focus_on(plt.gcf().axes, plt.gcf().axes[2])
# plt.savefig(f'../images/corr_beta={sim3.beta}.png')

# %% Show betas
beta1.corr(cut = 0.1)
# plt.savefig(f'../images/corr_beta={sim1.beta}.png')
beta2.corr(cut = 0.1)
# plt.savefig(f'../images/corr_beta={sim2.beta}.png')
beta3.corr(cut = 0.1)
# plt.savefig(f'../images/corr_beta={sim3.beta}.png')

# %%
