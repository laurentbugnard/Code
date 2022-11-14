#%% 
import numpy as np
import scipy.fftpack as fft
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from power_law_fit import power_law
from cplot2 import *

#%% 1D
L = 100

#generate power law in x
x = fft.fftfreq(L)*L #mais l'ordre n'import pas pour la ft d'après
f = power_law(np.abs(x), 1, 0.4)
f[f == float('inf')] = 1

#FT
q = fft.fftfreq(x.size)
f_t = fft.fft(f)

#Plot the F-Transforms
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
ax1.plot(x, f)

cplot2(ax2,q,f_t)

ax3.loglog(x[x>0], f[x>0])

cplot2(ax4,q[q>0],f_t[q>0], lognorm = True)
ax4.set_title(f'min at {q[np.argmin(np.abs(f_t))]:.2f}')
ax4.axvline(q[np.argmin(np.abs(f_t))], linestyle = '--', color = 'k')

#%% 2D
L = 100

#generate power law in x
x = fft.fftfreq(L)*L
y = x.copy()
x,y = np.meshgrid(x,y)
f = power_law(np.sqrt(x**2+y**2), 1, 3)
f[f == float('inf')] = 1


#FT
qx = fft.fftfreq(x.size)
qy = qx.copy()
f_t = fft.fft2(f)


plt.figure()
plt.subplot(2,2,1)
plt.imshow(f)
plt.subplot(2,2,2)
plt.imshow(np.real(f_t))
plt.title('REAL PART')
plt.subplot(2,2,3)
plt.imshow(f, norm = LogNorm())
plt.subplot(2,2,4)
plt.imshow(np.real(f_t), norm = LogNorm())
plt.title('REAL PART')
plt.colorbar()
