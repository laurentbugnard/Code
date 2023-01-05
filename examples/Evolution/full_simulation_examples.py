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
    'exponent': 0.8,
    'p': 0.1,
    'stabCoef': 2
}
nsteps=1000

sim_results = full_simulation(params, nsteps, seed=1)

#%% Static version
show_results(**sim_results)


#%% Animated version

anim = show_results(**sim_results, show_animation = True, fps=60, rate = 5)

#%%
#%% Write to video
import matplotlib as mpl 
mpl.rcParams['animation.ffmpeg_path'] = r"C:/Users/laure/Desktop/ffmpeg-2022-12-08-git-9ca139b2aa-essentials_build/bin/ffmpeg.exe"

writervideo = FFMpegWriter(fps=40) 
anim.save("./animation.mp4" , writer=writervideo)

# %%
