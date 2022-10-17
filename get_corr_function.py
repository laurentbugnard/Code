import numpy as np
from scipy import fft


def get_corr_function(s_t):
    K_map = fft.ifft2(np.abs(s_t)**2)
    K_map_centered = fft.fftshift(K_map)
    K = K_map_centered[K_map_centered.shape[0]//2,:]
    return K_map, K