from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft

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