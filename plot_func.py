from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft

def plot_map(f, name, centered = False):
    if(centered):
        f = fft.fftshift(f)
        plt.text(0,0,"centered", bbox={'facecolor': 'white', 'pad': 3})
    plt.imshow(f, origin = 'lower')
    plt.colorbar()
    plt.title(name)
    plt.xlabel('x')
    plt.ylabel('y')
    
def plot_ft(f, name, centered = True, log = False):
    
    N = f.shape[0]
    qx,qy = np.ogrid[-N/2:N/2,-N/2:N/2]
    
    xlabels = qx.ravel().copy()
    ylabels = qy.ravel().copy()
    
    if(centered):
        f = fft.fftshift(f)
    else:
        xlabels = np.concatenate((xlabels[xlabels >= 0], xlabels[xlabels < 0]))
        ylabels = np.concatenate((ylabels[ylabels >= 0], ylabels[ylabels < 0]))
    
    if(log):
        plt.imshow(np.abs(f), origin = 'lower', norm = LogNorm(vmax = np.max(np.abs(f))/1e3))
    else:
        plt.imshow(np.abs(f), origin = 'lower')
        
    plt.colorbar()
    plt.title(name)
    plt.gca().set_xticks(range(len(xlabels)))
    plt.gca().set_yticks(range(len(ylabels)))
    plt.gca().set_xticklabels(xlabels)
    plt.gca().set_yticklabels(ylabels)
    plt.locator_params(nbins=10)
    plt.xlabel('q_x')
    plt.ylabel('q_y')


def show_plots(u,C,s,u_t,C_t,s_t, N, xi, beta):
    
    plt.figure(figsize = (25,12), dpi = 80)
    plt.suptitle(f'N = {N}, xi = {xi}, beta = {beta}', fontsize = 30)
    
    plt.subplot(2,3,1)
    plot_map(u,'u')
    
    plt.subplot(2,3,2)
    plot_map(C.real,'C', centered = True)
    
    plt.subplot(2,3,3)
    plot_map(s.real,'s')
    
    
    
    plt.subplot(2,3,4)
    plot_ft(u_t,'u_t')
    
    plt.subplot(2,3,5)
    plot_ft(C_t,'C_t', log = True)
    
    plt.subplot(2,3,6)
    plot_ft(s_t,'s_t', log = True)

    plt.show()
    
    