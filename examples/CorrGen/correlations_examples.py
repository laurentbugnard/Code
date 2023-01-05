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
# sim = CorrGen(L = 1000, xi = float('inf'))

# alpha_working_1 = copy.copy(sim)
# alpha_working_1.generate_fields(method = 'alpha', exponent= 0.5)

# alpha_working_2 = copy.copy(sim)
# alpha_working_2.generate_fields(method = 'alpha', exponent= 0.8)

# alpha_working_3 = copy.copy(sim)
# alpha_working_3.generate_fields(method = 'alpha', exponent=1.2)

# alpha_limit_low_100 = CorrGen(L = 100, xi = float('inf'))
# alpha_limit_low_100.generate_fields(method = 'alpha', exponent= 0.1)

# alpha_limit_low_1000 = CorrGen(L = 1000, xi = float('inf'))
# alpha_limit_low_1000.generate_fields(method = 'alpha', exponent= 0.1)

# alpha_limit_low_5000 = CorrGen(L = 5000, xi = float('inf'))
# alpha_limit_low_5000.generate_fields(method = 'alpha', exponent= 0.1)

alpha_limit_high_2 = CorrGen(L = 1000, xi = float('inf'))
alpha_limit_high_2.generate_fields(method = 'alpha', exponent=2)
alpha_limit_high_3 = CorrGen(L = 1000, xi = float('inf'))
alpha_limit_high_3.generate_fields(method = 'alpha', exponent=3)
alpha_limit_high_4 = CorrGen(L = 1000, xi = float('inf'))
alpha_limit_high_4.generate_fields(method = 'alpha', exponent=4)


# %%
# alpha_working_1.measure_corr(scale='log', cut = 0.08)
# focus_on(plt.gcf().axes, [plt.gcf().axes[4], plt.gcf().axes[5], plt.gcf().axes[8]])
# plt.savefig(f'./examples/CorrGen/corr_alpha_working_1.png')
# alpha_working_2.measure_corr(scale='log', cut = 0.08)
# focus_on(plt.gcf().axes, [plt.gcf().axes[4], plt.gcf().axes[5], plt.gcf().axes[8]])
# plt.savefig(f'./examples/CorrGen/corr_alpha_working_2.png')
# alpha_working_3.measure_corr(scale='log', cut = 0.08)
# focus_on(plt.gcf().axes, [plt.gcf().axes[4], plt.gcf().axes[5], plt.gcf().axes[8]])
# plt.savefig(f'./examples/CorrGen/corr_alpha_working_3.png')

# alpha_limit_low_100.measure_corr(scale='log', cut = 0.1)
# focus_on(plt.gcf().axes, [plt.gcf().axes[4], plt.gcf().axes[5], plt.gcf().axes[7], plt.gcf().axes[8]])
# plt.savefig('./examples/CorrGen/corr_alpha_low_100.png')
# alpha_limit_low_1000.measure_corr(scale='log', cut = 0.1)
# focus_on(plt.gcf().axes, [plt.gcf().axes[4], plt.gcf().axes[5], plt.gcf().axes[7], plt.gcf().axes[8]])
# plt.savefig('./examples/CorrGen/corr_alpha_low_1000.png')
# alpha_limit_low_5000.measure_corr(scale='log', cut = 0.1)
# focus_on(plt.gcf().axes, [plt.gcf().axes[4], plt.gcf().axes[5], plt.gcf().axes[7], plt.gcf().axes[8]])
# plt.savefig('./examples/CorrGen/corr_alpha_low_5000.png')

alpha_limit_high_2.measure_corr(scale='log', cut = 0.015)
focus_on(plt.gcf().axes, [plt.gcf().axes[4], plt.gcf().axes[5], plt.gcf().axes[7], plt.gcf().axes[8]])
plt.savefig('./examples/CorrGen/corr_alpha_high_2.png')
alpha_limit_high_3.measure_corr(scale='log', cut = 0.015)
focus_on(plt.gcf().axes, [plt.gcf().axes[4], plt.gcf().axes[5], plt.gcf().axes[7], plt.gcf().axes[8]])
plt.savefig('./examples/CorrGen/corr_alpha_high_3.png')
alpha_limit_high_4.measure_corr(scale='log', cut = 0.015)
focus_on(plt.gcf().axes, [plt.gcf().axes[4], plt.gcf().axes[5], plt.gcf().axes[7], plt.gcf().axes[8]])
plt.savefig('./examples/CorrGen/corr_alpha_high_4.png')


# %%
# test = CorrGen(L=1000, xi=5)
# test.generate_fields(method = 'alpha', exponent= 0.8)
# test.measure_corr(cut=0.1, scale='log')



# %%
