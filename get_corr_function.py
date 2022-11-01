#ADD COMMENTS AND CORRECT

import numpy as np
from scipy import fft


def get_corr_function(f, full_map = False, normalized = True):
    #full complex correlation map
    f_t = fft.fft2(f) #gett the fourier transform
    K_map = fft.ifft2(np.abs(f_t)**2) / f.size #normalize by the number of points
    K_map_centered = fft.fftshift(K_map) #center to easily take a line
    K_line = K_map_centered[K_map_centered.shape[0]//2,:K_map_centered.shape[1]//2]
    K = np.flip(K_line)
    if(full_map):
        return K_map_centered
    else:
        if(normalized):
            return K.real / np.var(f)
        else:
            return K.real