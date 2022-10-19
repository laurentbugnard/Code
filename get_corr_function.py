import numpy as np
from scipy import fft


def get_corr_function(f_t):
    K_map = fft.ifft2(np.abs(f_t)**2) / f_t.size
    K_map_centered = fft.fftshift(K_map)
    K = K_map_centered[K_map_centered.shape[0]//2,:]
    return K.real