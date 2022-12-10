#%% Imports
from get_corr_function import *
from get_values import *
from config.ipy_config import *
from plot_func import *
from power_law_fit import *
from scan import *
from CorrGen import *
ipy_config()

#%% Generate 2 simulations
sim1 = CorrGen(L = 1000, xi = float('inf'), beta = 0.8)
sim1.generate_fields(s_centered = True, s_normalized= True)
sim2 = CorrGen(L = 1000, xi = float('inf'), beta = 0.3)
sim2.generate_fields(s_centered = True, s_normalized= True)
sim3 = CorrGen(L = 1000, xi = float('inf'), beta = 1.1)
sim3.generate_fields(s_centered = True, s_normalized= True)

# %%
sim1.corr(cut = 0.1)
plt.savefig(f'examples/corr_beta={sim1.beta}.png')
sim2.corr(cut = 0.1)
plt.savefig(f'examples/corr_beta={sim2.beta}.png')
sim3.corr(cut = 0.1)
plt.savefig(f'examples/corr_beta={sim3.beta}.png')

# %%
