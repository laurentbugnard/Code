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
from animation import *

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
sim_results = evolution_verbose(system, 1000)



#%%

anim = animation(sigmay_mean, propagator, **sim_results, frames_per_second=10, rate = 3)

# f = "./animation.mp4" 
# writervideo = FFMpegWriter(fps=60) 
# anim.save(f, writer=writervideo)
#TODO problem: FFMpegWriter needs to access get_size_inches of Figure but I'm using subfigures

# %%
