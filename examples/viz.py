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
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LogNorm
from show_results import *

#%%
L = 50
#TODO ask to have direct access to sigmay_mean, propagator, etc.

sigmay_mean=np.ones([L, L])
propagator, distances_rows, distances_cols = elshelby_propagator(L=L, imposed="strain")

system = SystemAthermal(
    propagator=propagator,
    distances_rows=distances_rows,
    distances_cols=distances_cols,
    sigmay_mean=sigmay_mean,
    sigmay_std=0.3 * np.ones([L, L]),
    seed=0,
    init_random_stress=False,
    init_relax=True,
    sigmabar=0
)

#%%
sim_results = evolution_verbose(system, 500)



#%%

anim = show_results(sigmay_mean, propagator, **sim_results, animate = True, fps=60, rate = 1)

#%%
import matplotlib as mpl 
mpl.rcParams['animation.ffmpeg_path'] = r"C:/Users/laure/Desktop/ffmpeg-2022-12-08-git-9ca139b2aa-essentials_build/bin/ffmpeg.exe"
#%%
writervideo = FFMpegWriter(fps=40) 
anim.save("./animation.mp4" , writer=writervideo)
#TODO problem: FFMpegWriter needs to access get_size_inches of Figure but I'm using subfigures

# %%
# show_results(sigmay_mean, propagator, **sim_results)