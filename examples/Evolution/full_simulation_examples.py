#%%
import os
os.chdir('C:/Users/laure/Documents/EPFL/MA/MA1/PCSL/EPM')
import sys
sys.path.append('./modules')
from config.ipy_config import ipy_config
ipy_config()

from EPM_func.EPM_func import *
from matplotlib.animation import FFMpegWriter
from Plotting.plot_func import *
from Simulation.simulate import *

#%% PARAMETERS TO SIMULATE
# params = {
#     'L': 100,
#     'xi': float('inf'),
#     'method': 'alpha',
#     'exponent': 0.01,
#     'p': 1,
#     'stabCoef': 1
# }


#%% Generate examples
params = {'L': 100, 'stabCoef': 1}
homog_failure, _ = simulate(params, nsteps=400, homogeneous=True)
params.update({'stabCoef': 2.5})
homog_flow, _ = simulate(params, nsteps=600, homogeneous=True)
params.update({
    'xi': float('inf'),
    'method': 'alpha',
    'exponent': 0.8,
    'p': 0.5,
    'stabCoef': 1
})
progressive_failure, CorrGen_params = simulate(params, nsteps=600)


#%% Plot results
# show_results(**homog_failure)
# plt.savefig(f'examples/Evolution/homog_failure.png')
# show_results(**homog_flow)
# plt.savefig(f'examples/Evolution/homog_flow.png')
show_results(**progressive_failure, CorrGen_params=CorrGen_params)
plt.savefig(f'examples/Evolution/progressive_failure.png')










#%% Animated version

# anim = show_results(**sim_results, show_animation = True, fps=60, rate = 3)

## Write to video
# import matplotlib as mpl 
# mpl.rcParams['animation.ffmpeg_path'] = r"C:/Users/laure/Desktop/ffmpeg-2022-12-08-git-9ca139b2aa-essentials_build/bin/ffmpeg.exe"

# writervideo = FFMpegWriter(fps=40) 
# anim.save("./animation.mp4" , writer=writervideo)

# %%
