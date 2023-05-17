from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from scipy.optimize import curve_fit
plt.style.use('./modules/config/style.mplstyle')
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LinearRegression
from matplotlib.widgets import TextBox, Button
from tqdm import tqdm
from linetimer import linetimer


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
        L = f.shape[0]
        xlabels = (np.fft.fftfreq(L)*L).astype(int)
        ylabels = xlabels.copy()
        xlabels.sort()
        ylabels.sort()
        
        plt.gca().set_xticks(range(len(xlabels)))
        plt.gca().set_yticks(range(len(ylabels)))
        plt.gca().set_xticklabels(xlabels)
        plt.gca().set_yticklabels(ylabels)
        plt.locator_params(nbins=10)
    
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
        xlabels.sort()
        ylabels.sort()
    
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
                 sigmay:list[np.ndarray], sigma:list[np.ndarray], epsp:list[np.ndarray], 
                 relax_steps:np.ndarray, shift:np.ndarray, failing:np.ndarray,
                 stability_hist_list:np.ndarray, stability_kde_x_list:np.ndarray, stability_kde_y_list:np.ndarray,
                 relax_steps_hist_list:np.ndarray,
                 stability_bins_edges:np.ndarray,
                 relax_steps_bins_edges:np.ndarray,
                 CorrGen_params = None,
                 show_animation = False, rate = 1, fps = 1,
                 cut = False,
                 epsp_scale='linear'):
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
        sigma (list[np.ndarray]): Unpacked from ``evolution_verbose``.
        epsp (list[np.ndarray]): Unpacked from ``evolution_verbose``.
        relax_steps (np.ndarray): Unpacked from ``evolution_verbose``.
        shift (np.ndarray): Unpacked from ``evolution_verbose``.
        failing (np.ndarray): Unpacked from ``evolution_verbose``.
        stability_hist_list (np.ndarray): Unpacked from ``evolution_verbose``.
        stability_kde_x_list (np.ndarray): Unpacked from ``evolution_verbose``.
        stability_kde_y_list (np.ndarray): Unpacked from ``evolution_verbose``.
        relax_steps_hist_list (np.ndarray): Unpacked from ``evolution_verbose``.
        CorrGen_params (dict, optional): Added in ``full_simulation``. Defaults to None.
        show_animation (bool, optional): Determines whether an animation or just the final result should be returned. 
        Defaults to False.
        rate (int, optional): Number of data points per frame for the animation. Defaults to 1.
        fps (int, optional): Frames per second for the animation. Defaults to 1.
        cut (bool, optional): If True, only the evolution until the largest avalanche is shown.
        epsp_scale (str, optional): The scale ('log' or 'linear') for the epsp image. Defaults to 'linear'.

    Returns:
        Either just a figure of the final results or an animation object (matplotlib.animation.FuncAnimation),
        depending on ``show_animation``.
    """
    
    
    ########################### Local functions and classes for animation ###################
    class IndexTracker:
        def __init__(self, data_size = float('inf')):
            self.index = 0 #initialize index
            # print('Index = ', self.index)
            self.data_size = data_size
            
        def set_index(self, index):
            self.index = index
            # print('Index = ', self.index)

        def on_scroll(self, event):
            # print(event.button, event.step)
            
            #update index
            if event.button == 'up': increment = - int(2**(event.step-1))  
            else: increment = int(2**(-event.step-1))
            self.update_index(increment)
            
            #update figure
            update_all_axes(self.index)
        
        def on_press(self, event):
            # print('press', event.key, flush=True)
            
            #update index
            if event.key == 'left': increment = -1
            elif event.key == 'shift+left': increment = -15
            elif event.key == 'right': increment = 1
            elif event.key == 'shift+right': increment = 15
            else: return
            
            self.update_index(increment, clip=False)
            
            #update figure
            update_all_axes(self.index)
            
        def on_mouse_click(self, event):
            self.set_index(int(event.xdata))
            update_all_axes(self.index)
            
        def update_index(self, increment, clip=True):
            if clip:
                self.index = np.clip(self.index + increment, 0, self.data_size -1) #stop at extreme i values
            else:
                self.index += increment
                self.index = self.index % self.data_size
            
            # print('Index = ', self.index)
            

    def animate(frame):
        update_all_axes(index=frame*rate)
    
    # @linetimer(unit='s')
    def update_all_axes(index):
        sigma_image.set_data(sigma[index])
        epsp_image.set_data(epsp[index])
    
        axes_plots[0].set_xlim(gammabar[0], gammabar[index]+0.01) #0.01 to avoid xlim = [0,0]
        # stress_strain.set_data(gammabar[0:index + 1], sigmabar[0:index + 1])
        stress_strain_extended.set_data(gammabar_extended[0:2*(index + 1)], sigmabar_extended[0:2*(index + 1)])
        axes_plots[1].set_ylim(0, np.max(relax_steps[0:index + 1]) + 1)
        avalanche_size.set_data(np.arange(index + 1), relax_steps[0:index + 1])

        events.set_data((epsp[index] - epsp[index-1])!=0)
        
        for count, rect in zip(stability_hist_list[index],
                               stability_bar_containers.patches):
            rect.set_height(count)
            
        stability_kde.set_data(stability_kde_x_list[index],stability_kde_y_list[index])
        
        for count, rect in zip(relax_steps_hist_list[index],
                               relax_steps_bar_containers.patches):
            rect.set_height(count)
        
        fig.canvas.draw()
    
    ############################### END OF LOCAL FUNCTIONS ######################################
    
    #extend stress-strain curve with shift
    gammabar_extended = extend_with_shift(gammabar, shift)
    sigmabar_extended = extend_with_shift(sigmabar, shift)
    
    if cut: last = np.argmax(relax_steps)
    else: last = -1
    
    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(2,2,wspace=0, hspace=0, width_ratios=[2,1])
    # plt.subplots_adjust(wspace=0.4, bottom=0.15)
    
    ###Images###
    subfigs[0,0].set_facecolor('0.95')
    axes_images = subfigs[0,0].subplots(1,2)
    #sigma(x)
    ax = axes_images[0]
    sigma_image = ax.imshow(sigma[last], vmin = np.min(sigma), vmax = np.max(sigma))
    sigma_cbar = subfigs[0,0].colorbar(sigma_image, aspect=10)
    ax.set_title(r'$\sigma(x)$')
    #epsp(x)
    ax = axes_images[1]
    if epsp_scale == 'log':
        epsp_image = ax.imshow(epsp[last], norm=LogNorm(vmin = np.min(epsp[last][epsp[last]!=0]), vmax = np.max(epsp[last])), interpolation='none')
    else:
        epsp_image = ax.imshow(epsp[last], vmin = np.min(epsp[last]), vmax = np.max(epsp[last]))
        
    epsp_cbar = subfigs[0,0].colorbar(epsp_image, aspect=10)
    ax.set_title(r'$\epsilon_p(x)$')

    ###Parameters###
    subfigs[0,1].set_facecolor('0.75')
    axes_parameters = subfigs[0,1].subplots(2,2)
    if CorrGen_params == None: subfigs[0,1].suptitle(f'L = {sigma[0].shape[0]}', fontsize=18)
    else:
        if CorrGen_params['method'] == 'alpha': 
            exponent_title = rf'$\alpha = {CorrGen_params["exponent"]}$, '
        elif CorrGen_params['method'] == 'beta':
            exponent_title = rf'$\beta = {CorrGen_params["exponent"]}$, '
        subfigs[0,1].suptitle(fr'L = {sigma[0].shape[0]}, $\xi$ = {CorrGen_params["xi"]}, ' +
                              exponent_title + f'p = {CorrGen_params["p"]}', fontsize=18)
    #sigmaY(x)
    ax = axes_parameters[0,0]
    sigmay_mean[sigmay_mean == float('inf')] = np.nan
    sigmaY_image = ax.imshow(sigmay_mean, interpolation=None)
    sigmaY_cbar = subfigs[0,1].colorbar(sigmaY_image, aspect=5)
    ax.set_title(r'$<\sigma^Y(x)>$', fontsize=15)
    #G(x)
    ax = axes_parameters[0,1]
    propagator_image = ax.imshow(propagator, norm = LogNorm())
    ax.set_xticks([])
    ax.set_yticks([])
    propagator_cbar = subfigs[0,1].colorbar(propagator_image, aspect = 5)
    ax.set_title(r'$G(x)$', fontsize=15)
    #Stability distribution
    ax = axes_parameters[1,0]
    
    _, _, stability_bar_containers = ax.hist([0], bins=stability_bins_edges,
                                              ec="black", alpha=0.5, density = True)
    for count, rect in zip(stability_hist_list[last], stability_bar_containers):
        rect.set_height(count)
        
    stability_kde = ax.plot(stability_kde_x_list[last], stability_kde_y_list[last])[0]
    ax.set_title(r'$P(x)$', fontsize=15)
    ax.set_xlim(stability_bins_edges[0],stability_bins_edges[-1])
    stability_bins_edges_width = stability_bins_edges[1] - stability_bins_edges[0]
    ax.set_ylim(0, 1/stability_bins_edges_width)
    #
    axes_parameters[1,1].axis('off')

    ###Plots###
    subfigs[1,0].set_facecolor('0.75')
    axes_plots = subfigs[1,0].subplots(1,2)

    ax = axes_plots[0]
    # stress_strain = ax.plot(gammabar[0:last], sigmabar[0:last])[0]
    stress_strain_extended = ax.plot(gammabar_extended[0:2*last], sigmabar_extended[0:2*last])[0]
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$\sigma$")

    ax = axes_plots[1]
    avalanche_size = ax.plot(relax_steps[0:last])[0]
    ax.set_xlabel("step")
    ax.set_ylabel("avalanche size")

    ###Avalanches###
    subfigs[1,1].set_facecolor('0.95')
    axes_avalanches = subfigs[1,1].subplots(1,2)

    ax = axes_avalanches[0]
    events = ax.imshow((epsp[last]-epsp[0])!=0, vmin=0, vmax=1)
    ax.set_title('Events')
    
    ax = axes_avalanches[1]
    _, _, relax_steps_bar_containers = ax.hist(relax_steps[1:last], bins=relax_steps_bins_edges,
                                              ec="black", alpha=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel('count', fontsize=12)
    ax.set_title('Avalanche statistics', fontsize=15)

    #maximize window
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()

    if(show_animation):
        # plt.close('all')
        return FuncAnimation(fig, animate, frames=int(np.floor(len(sigma)/rate)) -1 , interval= int(1/fps*1000))

    
    #connect figure to keyboard and use a tracker for the index
    global tracker #trick: make tracker global so it exists outside show_results and the connection remains
    tracker = IndexTracker(len(sigma))
    fig.canvas.mpl_connect('key_press_event', tracker.on_press) 
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    # fig.canvas.mpl_connect('button_press_event', tracker.on_mouse_click)
    
    #add textbox to navigate
    def submit(index):
        try:
            index = int(index)
            tracker.set_index(index)
            update_all_axes(np.clip(int(index), 0, len(sigma)))
        except:
            pass
        text_box.set_val('')
    
    boxsize = (0.04, 0.06)
    axbox = subfigs[0,0].add_axes([0, 1-boxsize[1], *boxsize])
    global text_box
    text_box = TextBox(axbox, 'step')
    text_box.on_submit(submit)
    
    #add button to change scale
    
    def scale_dynamic(event):
        sigma_image.set_clim(vmin=np.min(sigma[tracker.index]), vmax=np.max(sigma[tracker.index]))
        epsp_image.set_clim(vmin=np.min(epsp[tracker.index]), vmax=np.max(epsp[tracker.index]))
        fig.canvas.draw()
        
    def scale_last(event):
        sigma_image.set_clim(vmin=np.min(sigma[last]), vmax=np.max(sigma[last]))
        epsp_image.set_clim(vmin=np.min(epsp[last]), vmax=np.max(epsp[last]))
    
    buttonsize = (boxsize[0]*2, boxsize[1])
    
    axbutton_dynamic = subfigs[0,0].add_axes([0, 1-2*buttonsize[1], *buttonsize])
    global button_dynamic
    button_dynamic = Button(axbutton_dynamic, 'scale: dynamic')
    button_dynamic.on_clicked(scale_dynamic)
    
    axbutton_last = subfigs[0,0].add_axes([0, 1-3*buttonsize[1], *buttonsize])
    global button_last
    button_last = Button(axbutton_last, 'scale: last')
    button_last.on_clicked(scale_last)
    
    return fig


def set_alpha(ax, alpha):
    for spine in ax.spines.values():
        spine.set_color((0,0,0,alpha))
    ax.tick_params(axis="x", colors=(0,0,0,alpha))
    ax.tick_params(axis="y", colors=(0,0,0,alpha))
    ax.xaxis.label.set_color((0,0,0,alpha))
    ax.yaxis.label.set_color((0,0,0,alpha))

    for child in ax.get_children():
        try:
            child.set_alpha(alpha)
        except:
            pass
        
        
def focus_on(all_axes, focus_axes, alpha=0.3):
    
    if type(focus_axes) != list:
        focus_axes = [focus_axes]
    
    for ax in all_axes:
        if not(ax in focus_axes):
            set_alpha(ax, alpha)
            
def compute_and_show_statistics(epspbar, sigmabar, shift, epsp_map, 
                                eps_sample_start=0, n_bins=10, cut_at=1):
    
    #preparation
    eps = epspbar+sigmabar #TODO: change with gammabar
    eps = extend_with_shift(eps,shift)
    sigmabar = extend_with_shift(sigmabar,shift)
    
    
    sample_start = np.argwhere(eps>eps_sample_start)[0][0]
    
    fig, axes = plt.subplots(2,2, figsize=(15,9))
    
    axes[0,0].plot(eps, sigmabar)
    axes[0,0].axvline(eps[sample_start], color='red', linestyle='--', 
                    label=f'# of samples: {len(eps)-sample_start}')
    axes[0,0].set_xlabel(r'$\epsilon$')
    axes[0,0].set_ylabel(r'$\sigma$')
    axes[0,0].legend()
    
    
    epsp_image = axes[0,1].imshow(epsp_map, vmin = np.min(epsp_map), vmax = np.max(epsp_map))
    fig.colorbar(epsp_image, aspect=10)
    axes[0,1].set_title(r'$\epsilon_p(x)$')
        
    delta_sigmabar = np.diff(sigmabar)[sample_start:]
    unloadings = -1 * delta_sigmabar[delta_sigmabar<0]
    
    axes[1,0].hist(unloadings, bins=n_bins, density=True)
    axes[1,0].set_xlabel(r'$\Delta \sigma$')
    axes[1,0].set_ylabel('density')
    
    bins = np.logspace(np.log10(np.min(unloadings)),np.log10(np.max(unloadings)), n_bins)
    
    
    for x_coord in bins:
        axes[1,0].axvline(x_coord, color='k', linestyle='--')
    
    
    histogram = axes[1,1].hist(unloadings, bins=bins, density=True, edgecolor='k')
    
    axes[1,1].set_xscale('log')
    axes[1,1].set_yscale('log')
    axes[1,1].set_xlabel(r'$\Delta \sigma$')
    axes[1,1].set_ylabel('density')
    
    #Select fitting values and range
    centers = np.sqrt(bins[0:-1] * bins[1:]) #use geometric means for centers
    
    if type(cut_at) == list:
        centers = centers[cut_at[0]:cut_at[1]]
        values = histogram[0][cut_at[0]:cut_at[1]]
    else:
        last_index = int(cut_at*centers.size)
        centers = centers[0:last_index]
        values = histogram[0][0:last_index]
    
    axes[1,1].scatter(centers, values, color='red')
    
    #Do the power law fit:
    c, a = power_law_fit(centers, values)
    x_sample = np.linspace(centers[0], centers[-1], 10)
    axes[1,1].plot(x_sample, power_law(x_sample, c, a), 
                 linestyle = '--', color = 'k', label=f'Exponent: {a:.2f}')
    axes[1,1].legend()
    
    
def extend_with_shift(a, shift):
    a_extended = np.repeat(a,2)
    a_extended[1::2] += shift
    
    return a_extended