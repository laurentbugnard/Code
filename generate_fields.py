import numpy as np
from scipy import fft


def generate_u(N):
    u = np.random.randn(N,N)
    u_t = fft.fft2(u)
    return u,u_t

def generate_C(N, xi, beta):
    qx,qy = np.ogrid[-N/2:N/2,-N/2:N/2]
    normes = np.sqrt(qx**2 + qy**2)

    C_t_centered = 1/(normes**beta + xi**(-beta))

    C_t = fft.ifftshift(C_t_centered)
    C = fft.ifft2(C_t)
    return C, C_t