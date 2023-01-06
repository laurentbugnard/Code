#%%
import os
os.chdir('C:/Users/laure/Documents/EPFL/MA/MA1/PCSL/EPM')
import sys
sys.path.append('./modules')
from config.ipy_config import ipy_config
ipy_config()

import numpy as np
import matplotlib.pyplot as plt
from EPM_func.EPM_func import *
from matplotlib.animation import FFMpegWriter
from plot_func.plot_func import *
from full_func import full_simulation

#%% PARAMETERS TO SIMULATE OR IMPORT
params = {
    'L': 100,
    'xi': float('inf'),
    'method': 'alpha',
    'exponent': 0.01,
    'p': 1,
    'stabCoef': 1
}
nsteps=501

sim_results, CorrGen_params = full_simulation(params, nsteps, seed=2)

#%% Static version
show_results(**sim_results, CorrGen_params=CorrGen_params)



#%% Animated version

anim = show_results(**sim_results, show_animation = True, fps=60, rate = 3)

#%%
#%% Write to video
import matplotlib as mpl 
mpl.rcParams['animation.ffmpeg_path'] = r"C:/Users/laure/Desktop/ffmpeg-2022-12-08-git-9ca139b2aa-essentials_build/bin/ffmpeg.exe"

writervideo = FFMpegWriter(fps=40) 
anim.save("./animation.mp4" , writer=writervideo)

# %%
