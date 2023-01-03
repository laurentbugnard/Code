#%% Imports
import sys
sys.path.append('../../modules')
# from config.ipy_config import ipy_config
# ipy_config()
# from plot_func import *
import CorrGen

#%% Generate 2 simulations
sim1 = CorrGen(L = 100, xi = float('inf'), beta = 0.8)
sim1.generate_fields(s_centered = True, s_normalized= True)
sim1.generate_sigmaY(p = 0.1)

sim2 = CorrGen(L = 1000, xi = float('inf'), beta = 0.9)
sim2.generate_fields(s_centered = True, s_normalized= True)
sim2.generate_sigmaY(p = 0.1)
# %%
sim1.show_plots()
plt.savefig(f'examples/gen1.png')
sim2.show_plots()
plt.savefig(f'examples/gen2.png')

#%%
sim1.show_final()
plt.savefig(f'examples/fin1.png')
sim2.show_final()
plt.savefig(f'examples/fin2.png')
# %%