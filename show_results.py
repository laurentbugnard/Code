import numpy as np
import matplotlib.pyplot as plt
plt.style.use('../config/style.mplstyle')
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

def show_results(sigmay_mean, propagator, sigmabar, epspbar, gammabar, sigma, epsp, relax_steps, failing, animate = False, rate = 1, fps = 1):
    
    plt.close('all')
    fig = plt.figure()
    subfigs = fig.subfigures(2,2,wspace=0.3, width_ratios=[2,1])
    plt.subplots_adjust(wspace=0.4, bottom=0.15)

    ###Images###
    axes_images = subfigs[0,0].subplots(1,2)
    #sigma(x)
    ax = axes_images[0]
    sigma_image = ax.imshow(sigma[-1], vmin = -1.1, vmax = 1.1)
    sigma_cbar = subfigs[0,0].colorbar(sigma_image, aspect=10)
    ax.set_title(r'$\sigma(x)$')
    #epsp(x)
    ax = axes_images[1]
    epsp_image = ax.imshow(epsp[-1], vmin = 0, vmax = np.max(epsp[-1]))
    epsp_cbar = subfigs[0,0].colorbar(epsp_image, aspect=10)
    ax.set_title(r'$\epsilon_p(x)$')

    ###Parameters###
    axes_parameters = subfigs[0,1].subplots(1,2)
    subfigs[0,1].suptitle(f'L = {sigma[0].shape[0]}')
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
        axes_plots[1].set_ylim(0, np.max(relax_steps[0:frame*rate + 1]) + 1)
        avalanche_size.set_data(np.arange(frame*rate + 1), relax_steps[0:frame*rate + 1])

    if(animate):
        return FuncAnimation(fig, animate, frames=int(np.floor(len(sigma)/rate)) -1 , interval= int(1/fps*1000))
    
    return fig