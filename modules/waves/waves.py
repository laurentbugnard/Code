#%%
import os
os.chdir('C:/Users/laure/Documents/EPFL/MA/MA1/PCSL/EPM')
import sys
sys.path.append('./modules')
import numpy as np
import matplotlib.pyplot as plt
from config.ipy_config import ipy_config
ipy_config()

#%%
def create_wave(L, h, d, v, e_v=1, e_h=1, v_shift=50):

    lmbda = 2*(h-1) + 4*(d-1)
    A = 2*(v+d)
    
    assert L%lmbda == 0, 'Not valid'
    
    starts = np.array([lmbda*m for m in range(int(L/lmbda))])
    
    
    head = np.zeros((L,L))
    head[0,starts] = 1
    map = head.copy()
    for _ in range(1,h):
        head = np.roll(head, 1, axis=1)
        map += head
    for _ in range(1,d):
        head = np.roll(head, 1, axis=(0,1))
        map += head
    for _ in range(1,v):
        head = np.roll(head, 1, axis=0)
        map += head
    for _ in range(1,d-1):
        head = np.roll(head, 1, axis=(0,1))
        map += head
        
    map_flip = np.flip(map, axis=0)
    map_flip = np.roll(map_flip, v + 2*(d-1), axis=0)
    map_flip = np.roll(map_flip, h-1 + 2*(d-1), axis=1)

    map += map_flip
    head=map.copy()

    for _ in range(1,e_h):
        head = np.roll(head,1,axis=1)
        map += head
        
    head=map

    for _ in range(1,e_v):
        head = np.roll(head,1,axis=0)
        map += head
        
    map = np.roll(map, v_shift, axis=0)

    map[map>0]=1
    
    map_invert = map.copy()
    map_invert[map==1]=0
    map_invert[map==0]=1
    
    return map_invert


# %%
