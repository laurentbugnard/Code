#%%
import sys
sys.path.append('../')
from config.ipy_config import ipy_config
ipy_config()

from CorrGen import *


#%%
A = CorrGen(L=1000, xi= float('inf'))
A.generate_fields(method='alpha',exponent=0.3, reg_value=1)

# %%
A.corr(cut=0.1, C_corr=True)

# %%
