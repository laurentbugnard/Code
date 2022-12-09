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
from matplotlib.colors import LogNorm

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
sigmabar, epspbar, sigma, epsp, relax_steps, failing = evolution_verbose(system, 1000)
epsp
gammabar = epspbar + sigmabar

#%% All together (animated)
rate = 1
frames_per_second = 3
plt.close('all')
fig = plt.figure()
subfigs = fig.subfigures(2,2,wspace=0.3, width_ratios=[2,1])
plt.subplots_adjust(wspace=0.4, bottom=0.15)

###Images###
axes_images = subfigs[0,0].subplots(1,2)
#sigma(x)
ax = axes_images[0]
sigma_image = ax.imshow(sigma[0], vmin = -1.1, vmax = 1.1)
sigma_cbar = subfigs[0,0].colorbar(sigma_image, aspect=10)
ax.set_title(r'$\sigma(x)$')
#epsp(x)
ax = axes_images[1]
epsp_image = ax.imshow(epsp[0], vmin = 0, vmax = np.max(epsp[-1]))
epsp_cbar = subfigs[0,0].colorbar(epsp_image, aspect=10)
ax.set_title(r'$\epsilon_p(x)$')

###Parameters###
axes_parameters = subfigs[0,1].subplots(1,2)
subfigs[0,1].suptitle(f'L = {L}')
#sigmaY(x)
ax = axes_parameters[0]
sigmaY_image = ax.imshow(sigmay_mean)
sigmaY_cbar = subfigs[0,1].colorbar(sigmaY_image, aspect=5)
ax.set_title(r'$<\sigma^Y(x)>$')
#G(x)
ax = axes_parameters[1]
propagator_image = ax.imshow(propagator, norm = LogNorm())
propagator_cbar = subfigs[0,1].colorbar(propagator_image, aspect = 5)
ax.set_title(r'$G(x)$')

###Plots###
axes_plots = subfigs[1,0].subplots(1,2)

ax = axes_plots[0]
stress_strain = ax.plot(gammabar, sigmabar)[0]
ax.set_xlabel(r"$\epsilon$")
ax.set_ylabel(r"$\sigma$")

ax = axes_plots[1]
avalanche_size = ax.plot(relax_steps)[0]
ax.set_xlabel("step")
ax.set_ylabel("avalanche size")

###Avalanches###
axes_avalanches = subfigs[1,1].subplots(1,2)

ax = axes_avalanches[0]
ax.plot([])
ax.set_xlabel(r"step")
ax.set_ylabel(r"$\epsilon_{av}$")

ax = axes_avalanches[1]
ax.plot([])
ax.set_xlabel("step")
ax.set_ylabel("unstable particles")

#maximize window
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()



def animate(frame):

    sigma_image.set_data(sigma[frame*rate])
    epsp_image.set_data(epsp[frame*rate])
    
    axes_plots[0].set_xlim(gammabar[0], gammabar[frame*rate + 1])
    stress_strain.set_data(gammabar[0:frame*rate + 1], sigmabar[0:frame*rate + 1])
    avalanche_size.set_data(np.arange(frame*rate + 1), relax_steps[0:frame*rate + 1])

animation = FuncAnimation(subfigs[0,0], animate, frames=int(np.floor(len(sigma)/rate)), interval= int(1/frames_per_second*1000))



# #%% Image box

# fig, axes = plt.subplots(ncols=2)

# ax = axes[0]
# cax = ax.imshow(sigma[-1])
# ax.set_title(r'$\sigma(x)$')
# # cbar = fig.colorbar(cax)

# ax = axes[1]
# ax.imshow(epsp[-1])
# ax.set_title(r'$\epsilon_p(x)$')

# plt.show()
# #%% Evolution box

# fig, axes = plt.subplots(ncols=2)
# fig.tight_layout

# ax = axes[0]
# ax.plot(sigmabar + epspbar, sigmabar)
# ax.set_xlabel(r"$\epsilon$")
# ax.set_ylabel(r"$\sigma$")

# ax = axes[1]
# ax.plot(relax_steps)
# ax.set_xlabel(r"step")
# ax.set_ylabel(r"$relaxation steps$")

# plt.show()

# #%% Parameter box

# fig, axes = plt.subplots(1,2)
# fig.suptitle(f'L = {L}')

# ax = axes[0]
# ax.imshow(sigmay_mean)
# ax.set_title(r'$\sigma(x)$')

# ax = axes[1]
# ax.imshow(propagator)
# ax.set_title(r'$\epsilon_p(x)$')


# # %% Avalanches box



# # %%

# %%
