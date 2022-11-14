#ADD COMMENTS AND CORRECT

import numpy as np
from scipy import fft


def get_corr_function(f, full_map = False, normalized = True):
    f = f.real #first make sure it's real, because parasite complex values can change everything
    #full complex correlation map
    f_t = fft.fft2(f) #get the fourier transform
    #normalize by the number of points: shouldn't be necessary since ifft2 already does
    #don't know why, but it doesn't work without it:
    K_map = fft.ifft2(np.abs(f_t)**2)/f.size 
    K = K_map[0,:K_map.shape[1]//2]
    if(full_map):
        return K_map
    else:
        if(normalized):
            return K.real / np.var(f)
        else:
            return K.real