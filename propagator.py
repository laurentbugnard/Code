#%% 
import numpy as np
import scipy.fftpack as fft
from plot_func import *
import pandas as pd
import h5py

from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("matplotlib", "qt")
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

#%% Me

def propagator(L):
    qx = fft.fftfreq(L) *2 * np.pi
    qy = qx.copy()
    qx, qy = np.meshgrid(qx,qy)

    a = 2-2*np.cos(qx)
    b = 2-2*np.cos(qy)

    G_t = -4*a*b / (a+b)**2
    G_t[0,0] = 0 # TODO or -1?  verify

    G = fft.ifft2(G_t)
    return G.real


#%% Export
f = h5py.File('data.hdf5','r+')
for L in [30,50]:
    f.create_dataset(f'propagators/propL={L}', data = propagator(L))
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
