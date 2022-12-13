from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from scipy.optimize import curve_fit
plt.style.use('../config/style.mplstyle')
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LinearRegression

#############IMSHOW CONFIGURATION#################

def plot_map(f:np.ndarray, name:str, centered = False):
    """Plots a map in real space, possibly centered.

    Args:
        f (np.ndarray): Field that should be mapped.
        name (str): Title that should appear.
        centered (bool, optional): Determines whether the origin (0,0) should be set at the center of the image.
        Defaults to False.
    """
    
    if(centered):
        f = fft.fftshift(f)
        plt.text(0,0,"centered", bbox={'facecolor': 'white', 'pad': 3})
    plt.imshow(f, origin = 'lower')
    plt.colorbar()
    plt.title(name, fontsize = 20)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    
def plot_ft(f:np.ndarray, name:str, centered = True, logscale = False):
    """Plots a map (``f``) that is a 2D Fourier transform (of another map), possibly centered and using a logscale colormap.
    
    Args:
        f (np.ndarray): Field that should be mapped.
        name (str): Title that should appear.
        centered (bool, optional): Determines whether the origin (0,0) should be set at the center of the image. 
        Defaults to True.
        logscale (bool, optional): Determines whether a logscale colormap should be used. Defaults to False.
    """
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
        plt.imshow(np.abs(f), origin = 'lower', norm = LogNorm())
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
    
def regularized_power_law(x:np.ndarray, c:float, a:float, reg_value = 0.0) -> np.ndarray:
    """Power law with negative exponent: :math:`c x^{-a}`. The singularity at the origin is replaced by 0.
    This is necessary, as we are working with Fourier-Transforms, which do not accept 'inf' values in the inputs.

    Args:
        x (np.ndarray): Function input.
        c (float): Multiplicative constant.
        a (float): Exponent.
        reg_value (float): Value to impose at x = 0.

    Returns:
        np.ndarray: Regularized power law of ``x``.
    """
    
    f = c*(1/x)**a
    f[f == float('inf')] = reg_value
    #TODO check what we should impose at 0
    return f

def power_law(x:np.ndarray, c:float, a:float) -> np.ndarray:
    """Power law with negative exponent: :math:`c x^{-a}`.

    Args:
        x (np.ndarray): Function input.
        c (float): Multiplicative constant.
        a (float): Exponent.

    Returns:
        np.ndarray: Power law of ``x``.
    """
    return c*(1/x)**a

def power_law_fit(x:np.ndarray,y:np.ndarray) -> tuple[float,float]:
    """Fits a power law (negative exponent) to ``x`` and ``y`` by linear regression on the log-log plot. 
    Returns the optimal parameters ``c`` and ``a``.

    Args:
        x (np.ndarray): x-axis.
        y (np.ndarray): y-axis.

    Returns:
        float: Multiplicative constant.
        float: Exponent (negative).
    """
    
    #filter out 0 and negative values for which the logarithm can not be taken
    y_filtered = y[(x>0) & (y>0)]
    x_filtered = x[(x>0) & (y>0)]
    
    lx = np.log(x_filtered)
    ly = np.log(y_filtered)
    lin_reg = LinearRegression().fit(lx.reshape(-1,1),ly)
    
    return np.exp(lin_reg.intercept_), -lin_reg.coef_[0]

def plot_power_law_fit(x:np.ndarray, y:np.ndarray, label = "") -> float:
    """Generates a power law fit using ``power_law_fit`` and plots the result.

    Args:
        x (np.ndarray): x-axis.
        y (np.ndarray): y-axis.
        label (str, optional): Label for the plot. Defaults to "".

    Returns:
        float: Exponent (negative).
    """

    
    popt = power_law_fit(x, y)
    plt.scatter(x, y, label = label)
    plt.plot(x, power_law(x, *popt), color = 'k', label = f'|slope| = {popt[1]}')
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.legend()
    
    return popt[1]

############COMPLEX PLOTS#################
def cplot2(ax:plt.Axes, x:np.ndarray, f:np.ndarray, method = 'real-imag', lognorm = False):
    """Visualization function for a complex function ``f`` of ``x``. Always plots two curves:
    either the real and imaginary part (if ``method`` is 'real-imag') or the norm and the angle (if ``method`` is 'norm-arg').

    Args:
        ax (plt.Axes): Axes object on which we should plot.
        x (np.ndarray): x-axis.
        f (np.ndarray): complex function.
        method (str, optional): Either 'real-imag' or 'norm-arg'. Defaults to 'real-imag'.
        lognorm (bool, optional): Determines if the plots should be log-log. Defaults to False.
    """
    
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


def show_results(sigmay_mean:np.ndarray, propagator:np.ndarray, 
                 sigmabar:np.ndarray, epspbar:np.ndarray, gammabar:np.ndarray, 
                 sigma:list[float], epsp:list[float], 
                 relax_steps:np.ndarray, failing:np.ndarray, 
                 show_animation = False, rate = 1, fps = 1):
    """Visualization function which shows the whole evolution of the EPM. 
    Top-left pannel: Evolution of stress and plastic strain fields.
    Bottom-left pannel: Evolution of stress-strain curve and avalanche sizes.
    Top-right pannel: Parameters of the EPM.
    Bottom-right pannel: Avalanche analyzer (WIP).

    Args:
        sigmay_mean (np.ndarray): Unpacked from ``evolution_verbose``.
        propagator (np.ndarray): Unpacked from ``evolution_verbose``.
        sigmabar (np.ndarray): Unpacked from ``evolution_verbose``.
        epspbar (np.ndarray): Unpacked from ``evolution_verbose``.
        gammabar (np.ndarray):Unpacked from ``evolution_verbose``.
        sigma (list[float]): Unpacked from ``evolution_verbose``.
        epsp (list[float]): Unpacked from ``evolution_verbose``.
        relax_steps (np.ndarUnpacked from ``evolution_verbose``.
        failing (np.ndarray): Unpacked from ``evolution_verbose``.
        show_animation (bool, optional): Determines whether an animation or just the final result should be returned. 
        Defaults to False.
        rate (int, optional): Number of data points per frame for the animation. Defaults to 1.
        fps (int, optional): Frames per second for the animation. Defaults to 1.

    Returns:
        Either just a figure of the final results or an animation object (matplotlib.animation.FuncAnimation),
        depending on ``show_animation``.
    """
    
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