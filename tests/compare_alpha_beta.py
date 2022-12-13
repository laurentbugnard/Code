#%%
from CorrGen import CorrGen
from config.ipy_config import *
ipy_config()

#%%
beta = 1.2
alpha = 2*(1-beta)

simalpha = CorrGen(L=100, xi = float('inf'))
simalpha.generate_fields(method='alpha', exponent=alpha)
simalpha.generate_sigmaY(p=0.1)

simbeta = CorrGen(L=100, xi = float('inf'))
simbeta.generate_fields(method='beta', exponent=beta)
simbeta.generate_sigmaY(p=0.1)
#%%
simalpha.show_final()
simbeta.show_final()

# %%



#%% TEST DIFFERENT ALPHA: 0, 0.01, 0.1, 1, 10, 100000000
simalpha = CorrGen(L=100, xi = float('inf'))
simalpha.generate_fields(method='alpha', exponent=1e15)
simalpha.generate_sigmaY(p=0.1)

simalpha.show_final()


# %%
