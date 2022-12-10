from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from scipy.optimize import curve_fit
plt.style.use('../config/style.mplstyle')
from matplotlib.animation import FuncAnimation

#############IMSHOW CONFIGURATION#################

def plot_map(f, name, centered = False):
    '''Plots a map in real space, possibly centered'''
    
    if(centered):
        f = fft.fftshift(f)
        plt.text(0,0,"centered", bbox={'facecolor': 'white', 'pad': 3})
    plt.imshow(f, origin = 'lower')
    plt.colorbar()
    plt.title(name, fontsize = 20)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    
def plot_ft(f, name, centered = True, logscale = False):
    ''' Plots the 2d Fourier transform of a map. 
    The grid of frequencies is from -L/2 to L/2 in steps of 1.
    Possible to center the transform and set the colormap to logscale'''
    
    #prepare frequency grid
    L = f.shape[0]
    qx,qy = np.fft.fftfreq(L), np.fft.fftfreq(L)
    
    #get the labels for frequency axis (centered labels)
    xlabels = qx.ravel().copy()
    ylabels = qy.ravel().copy()
    
    #center if asked
    if(centered):
        f = fft.fftshift(f)
    else:
        #if not centered, make the label not centered
        xlabels = np.concatenate((xlabels[xlabels >= 0], xlabels[xlabels < 0]))
        ylabels = np.concatenate((ylabels[ylabels >= 0], ylabels[ylabels < 0]))
    
    #scale of the colormap
    if(logscale):
        plt.imshow(np.abs(f), origin = 'lower', norm = LogNorm())#(vmax = np.max(np.abs(f))/1e3))
    else:
        plt.imshow(np.abs(f), origin = 'lower')
        
    plt.colorbar()
    plt.title(name, fontsize = 20)
    #correct frequency axis labels
    plt.gca().set_xticks(range(len(xlabels)))
    plt.gca().set_yticks(range(len(ylabels)))
    plt.gca().set_xticklabels(xlabels)
    plt.gca().set_yticklabels(ylabels)
    plt.locator_params(nbins=10)
    plt.xlabel(r'$q_x$')
    plt.ylabel(r'$q_y$')
    
############POWER LAW FITS#################
    
def reg_power_law(x, c, a):
    '''Decreasing power law: c * x^(-a)'''
    f = c*(1/x)**a
    f[f == float('inf')] = 0
    return f

def power_law(x, c, a):
    '''Decreasing power law: c * x^(-a)'''
    return c*(1/x)**a

def power_law_fit(x,y):
    '''Function to fit a power law and return c(prefactor) and a(decrease exponent)'''

    popt, pcov = curve_fit(power_law,x,y)
    return popt

def plot_power_law_fit(x,y, label = ""):
    
    popt = power_law_fit(x, y)
    plt.scatter(x, y, label = label)
    plt.plot(x, power_law(x, *popt), color = 'k', label = f'|slope| = {popt[0]}')
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.legend()

    return popt[0]

############COMPLEX PLOTS#################
def cplot2(ax, x, f, method = 'real-imag', lognorm = False):
    if((method == 'real-imag') & (lognorm == False)):
        ax.plot(x, f.real, color = 'blue', label = 'Re')
        ax.plot(x, f.imag, color = 'red', label = 'Im')
        ax.legend()
    
    elif((method == 'real-imag') & (lognorm == True)):
        ax_right = ax.twinx()
        ax.plot(x, f.real, color = 'blue', label = 'Re')
        ax_right.plot(x, f.imag, color = 'red', label = 'Im')
        ax.legend()
        ax_right.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax_right.set_xscale('log')
        ax_right.set_yscale('log')
        
    
    elif(method == 'norm-arg'):
        ax_right = ax.twinx()
        ax.plot(x, np.abs(f), color = 'black', label = '||')
        ax_right.plot(x, np.angle(f), color = 'pink', label = 'angle')
        ax.legend()
        ax_right.legend()
        if(lognorm): 
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax_right.set_xscale('log')
            ax_right.set_yscale('log')

################ANIMATION#######################


def show_results(sigmay_mean, propagator, sigmabar, epspbar, gammabar, sigma, epsp, relax_steps, failing, show_animation = False, rate = 1, fps = 1):
    
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

    if(show_animation):
        return FuncAnimation(fig, animate, frames=int(np.floor(len(sigma)/rate)) -1 , interval= int(1/fps*1000))
    
    return fig