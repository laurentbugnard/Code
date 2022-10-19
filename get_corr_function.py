import numpy as np
from scipy import fft


def get_corr_function(f_t, full_map = False, normalized = False):
    K_map = fft.ifft2(np.abs(f_t)**2) / f_t.size
    K_map_centered = fft.fftshift(K_map)
    K_line = K_map_centered[K_map_centered.shape[0]//2,:K_map_centered.shape[1]//2]
    K = np.flip(K_line)
    K_normalized = K.real/K.real[0]
    if(full_map):
        return K_map_centered
    else:
        if(normalized):
            return K_normalized
        else:
            return K.real