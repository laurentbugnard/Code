#%%
import sys
sys.path.append('../') #include parent folder in the path
import numpy as np
from GooseEPM import elshelby_propagator
from GooseEPM import SystemAthermal
import matplotlib.pyplot as plt
from evolution import *
plt.style.use('../config/style.mplstyle')
from ipy_config import *
ipy_config()
from matplotlib.animation import FFMpegWriter
from show_results import *
import h5py

#%% Manually simulate
# L = 100
# #TODO ask to have direct access to sigmay_mean, propagator, etc.

# sigmay_mean=np.ones([L, L])
# propagator, distances_rows, distances_cols = elshelby_propagator(L=L, imposed="strain")

# system = SystemAthermal(
#     propagator=propagator,
#     distances_rows=distances_rows,
#     distances_cols=distances_cols,
#     sigmay_mean=sigmay_mean,
#     sigmay_std=0.00 * np.ones([L, L]),
#     seed=0,
#     init_random_stress=False,
#     init_relax=True,
#     sigmabar=0
# )

# #%%
# sim_results = evolution_verbose(system, 1000)
# sim_results.update({'sigmay_mean':sigmay_mean, 'propagator':propagator})

#%% Import data function
def import_data(group_name):
    f = h5py.File('../data/sim_results.hdf5','r')

    #get data as a hdf5 group
    data = f.get(group_name)

    #turn it back into a dictionary
    sim_results = {}
    for key in data.keys():
        sim_results.update({key: np.array(data[key])})
    
    f.close()
    
    return sim_results

#%% Import data
sim_results = import_data('sim_results_alpha=0.6')

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
