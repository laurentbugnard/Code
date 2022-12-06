#%% 
import numpy as np
import scipy.fftpack as fft
from plot_func import *
import pandas as pd
import h5py
from ipy_config import*
ipy_config()

#%% Me

def propagator(L, method = 'rossi'):
    qx = fft.fftfreq(L) *2 * np.pi
    qy = qx.copy()
    qx, qy = np.meshgrid(qx,qy)

    a = 2-2*np.cos(qx)
    b = 2-2*np.cos(qy)

    #G_E = Eshelby
    G_E_t = -4*a*b / (a+b)**2
    if (method == 'marko'):
        G_t = G_E_t
        G_t[0,0] = 0
    
    if (method == 'rossi'):
        G_t = G_E_t
        G_t[0,0] = 0 #to avoid nan in 0
        g = -1/(L**2 -1 ) *(np.sum(G_t) - G_t[0,0])
        G_t /= g
        G_t[0,0] = -1        

    G = fft.ifft2(G_t)
    return G.real

#%% Export
L_list = [20,30,50,100,150,200,1000]

f = h5py.File('data/data.hdf5','r+')
for L in L_list:
    if(f'/propagators/marko_propL={L}' in f):
        del f[f'propagators/marko_propL={L}']
    if(f'/propagators/rossi_propL={L}' in f):
        del f[f'propagators/rossi_propL={L}']
    f.create_dataset(f'propagators/marko_propL={L}', data = propagator(L, method='marko'))
    f.create_dataset(f'propagators/rossi_propL={L}', data = propagator(L, method='rossi'))
    
f.close()








#%%

G = propagator(64)

G_centered = fft.ifftshift(G)


# %% Marko to compare
####################################################################
#%%
df = pd.read_csv("marko_propagators/testing_propagator32_cosine.txt")
N = df["rx"].size
L = np.sqrt(N)

if(L - np.floor(L) != 0):
    print("Not a square map")

else:
    L = int(L)

    df_stack = np.stack([df["rx"],df["ry"],df["Greal"],df["Gimag"]])
    df_reshaped = df_stack.reshape(4,L,L)

    rx = df_reshaped[0,:,:]
    ry = df_reshaped[1,:,:]
    Greal = df_reshaped[2,:,:]
    Gimag = df_reshaped[3,:,:]







# %%
############################################################
plt.figure(figsize = (20,10), dpi = 80)
plt.subplot(1,3,1)
plt.title('Me')
plt.contourf(G_centered)
plt.axis('equal')
plt.colorbar()

plt.subplot(1,3,2)
plt.title('Marko')
plt.contourf(Greal)
plt.axis('equal')
plt.colorbar()


diff = Greal-G_centered

plt.subplot(1,3,3)
plt.title('Difference')
plt.contourf(diff)
plt.axis('equal')
plt.colorbar()
# %%
