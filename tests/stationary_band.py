#%%
from GooseEPM import SystemAthermal
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

#%% define
f = h5py.File('../data/data.hdf5','r')
propagator = np.array(f.get(f'propagators/rossi_propL=50')).real
propagator = propagator.copy()
L = propagator.shape[0]

system = SystemAthermal(
    propagator = propagator,
    distances_rows = np.fft.fftfreq(L)*L,
    distances_cols = np.fft.fftfreq(L)*L,
    sigmay_mean = np.ones_like(propagator),
    sigmay_std = np.ones_like(propagator)*0.0,
    init_random_stress = False,
    seed = 123
)

f.close()

#%% modify
band_location = int(L/2)

new_sigma = np.zeros_like(system.sigma)
bias = 1.2
fraction = 1
cut_off = int(L*fraction)
new_sigma[0:cut_off,band_location] = bias
# new_sigma[0:cut_off,band_location + 1] = -bias
system.sigma = new_sigma

#%% evolution
swap = False

maps = [system.sigma.copy()]
n_steps = 10000
for i in range(n_steps):
    if(np.all(np.abs(system.sigma) < 1)):
        print(f'converged after {i}/{n_steps}steps')
        break
    system.makeAthermalFailureStep()
    maps.append(system.sigma.copy())
    if(swap):
        j = np.argmax(system.sigma[:,band_location])
        k = np.random.choice(L)
        exchanged = system.sigma
        exchanged[[j,k], [band_location,band_location]] = exchanged[[k,j], [band_location,band_location]]
        system.sigma = exchanged

# %% Animate
speed = 1

plt.close('all')

fig = plt.figure(figsize=(15,8))
plt.subplot(1,3,1)
image = plt.imshow(maps[0], vmin = np.min(maps[-1]), vmax = np.max(maps[-1]))
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(maps[-1])
plt.colorbar()

plt.subplot(1,3,3)
plt.plot([np.mean(maps[i]) for i in range(len(maps))])
plt.title(r'$<\sigma>$', fontsize = 30)


def animate(frame):
    image.set_data(maps[frame*speed])

animation = FuncAnimation(fig, animate, frames=100, interval=100)
# %%
