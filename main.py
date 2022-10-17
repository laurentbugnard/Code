import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
#%% Plot functions
def plot_map(f, name):
    plt.imshow(f, origin = 'lower')
    plt.colorbar()
    plt.title(name)
    plt.xlabel('x')
    plt.ylabel('y')
    
def plot_ft(f, name, centered = True):
    
    xlabels = qx.ravel().copy()
    ylabels = qy.ravel().copy()
    
    if(centered):
        f = fft.fftshift(f)
    else:
        xlabels = np.concatenate((xlabels[xlabels >= 0], xlabels[xlabels < 0]))
        ylabels = np.concatenate((ylabels[ylabels >= 0], ylabels[ylabels < 0]))
    
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
    

#%% Show plots
def show_plots():
    
    plt.figure(figsize = (25,12), dpi = 80)
    
    plt.subplot(2,4,1)
    plot_map(u,'u')
    
    plt.subplot(2,4,2)
    plot_map(C.real,'C')
    
    plt.subplot(2,4,3)
    plot_map(s.real,'s')
    
    plt.subplot(2,4,4)
    plot_map(sigma.real, 'sigma')
    
    
    
    plt.subplot(2,4,5)
    plot_ft(u_t,'u_t')
    
    plt.subplot(2,4,6)
    plot_ft(C_t,'C_t')
    
    plt.subplot(2,4,7)
    plot_ft(s_t,'s_t')
    
    
#%% Computations
N = 100
xi = 0.1

np.random.seed(1)
u = np.random.randn(N,N)
u_t = fft.fft2(u)


qx,qy = np.ogrid[-N/2:N/2,-N/2:N/2]
normes2 = qx**2 + qy**2

C_t_centered = 1/(normes2 + xi**(-2))

C_t = fft.ifftshift(C_t_centered)

C = fft.ifft2(C_t)

s_t = C_t * u_t

s = fft.ifft2(s_t)

sigma = np.exp(s)

#%%
show_plots()