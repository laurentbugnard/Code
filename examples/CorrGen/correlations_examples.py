#%% Imports
import sys
import copy
sys.path.append('../')
from config.ipy_config import *
from plot_func import *
from CorrGen import *
ipy_config()

#%% Generate 2 simulations
sim = CorrGen(L = 1000, xi = float('inf'))

sim1 = copy.copy(sim)
sim1.generate_fields(method = 'beta', exponent= 0.8)

sim2 = copy.copy(sim)
sim2.generate_fields(method = 'beta', exponent= 0.3)

sim3 = copy.copy(sim)
sim3.generate_fields(method = 'beta', exponent= 1.1)

# %%
sim1.corr(cut = 0.1)
plt.savefig(f'../images/corr_beta={sim1.beta}.png')
sim2.corr(cut = 0.1)
plt.savefig(f'../images/corr_beta={sim2.beta}.png')
sim3.corr(cut = 0.1)
plt.savefig(f'../images/corr_beta={sim3.beta}.png')

# %%
