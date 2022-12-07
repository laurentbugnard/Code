#%%
import numpy as np
from GooseEPM import elshelby_propagator
from GooseEPM import SystemAthermal
import matplotlib.pyplot as plt
from evolution import *
plt.style.use('./config/style.mplstyle')

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


#%% Image box

fig, axes = plt.subplots(ncols=2)

ax = axes[0]
cax = ax.imshow(sigma[-1])
ax.set_title(r'$\sigma(x)$')
# cbar = fig.colorbar(cax)

ax = axes[1]
ax.imshow(epsp[-1])
ax.set_title(r'$\epsilon_p(x)$')

plt.show()
#%% Evolution box

fig, axes = plt.subplots(ncols=2)
fig.tight_layout

ax = axes[0]
ax.plot(sigmabar + epspbar, sigmabar)
ax.set_xlabel(r"$\epsilon$")
ax.set_ylabel(r"$\sigma$")

ax = axes[1]
ax.plot(relax_steps)
ax.set_xlabel(r"step")
ax.set_ylabel(r"$relaxation steps$")

plt.show()

#%% Parameter box

fig, axes = plt.subplots(1,2)
fig.suptitle(f'L = {L}')

ax = axes[0]
ax.imshow(sigmay_mean)
ax.set_title(r'$\sigma(x)$')

ax = axes[1]
ax.imshow(propagator)
ax.set_title(r'$\epsilon_p(x)$')


# %% Avalanches box


#%% All together
fig = plt.figure()
subfigs = fig.subfigures(2,2,wspace=0.3, width_ratios=[2,1])

#Images
axes_images = subfigs[0,0].subplots(1,2)

ax = axes_images[0]
ax.imshow(sigma[-1])
ax.set_title(r'$\sigma(x)$')

ax = axes_images[1]
ax.imshow(epsp[-1])
ax.set_title(r'$\epsilon_p(x)$')

#Parameters
axes_parameters = subfigs[0,1].subplots(1,2)

ax = axes_parameters[0]
ax.imshow(sigmay_mean)
ax.set_title(r'$\sigma(x)$')

ax = axes_parameters[1]
ax.imshow(propagator)
ax.set_title(r'$\epsilon_p(x)$')

#Plots
axes_plots = subfigs[1,0].subplots(1,2)

ax = axes_plots[0]
ax.plot(sigmabar + epspbar, sigmabar)
ax.set_xlabel(r"$\epsilon$")
ax.set_ylabel(r"$\sigma$")

ax = axes_plots[1]
ax.plot(relax_steps)
ax.set_xlabel(r"step")
ax.set_ylabel(r"$relaxation steps$")

#Avalanches
axes_avalanches = subfigs[1,1].subplots(1,2)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

# %%
