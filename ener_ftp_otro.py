import h5py 
from tqdm import tqdm
import numpy as np
from time import time
#%%
t1 = time()
na = 10
ali = np.zeros((1004,1004,na)) + 1j *  np.zeros((1004,1004,na))
with h5py.File('alturas.hdf5', 'r') as f:
    gdp = f.get('hs')
    alturas = gdp['alts']
    for i, ñ in zip(range(na), tqdm(range(na))):
        loll = alturas[i][10:-10,10:-10]
        ali[:,:,i] = np.fft.fft2(loll)
with h5py.File('alturas.hdf5', 'w') as f:
    fourier_im = f.create_group('fi')
    transs = fourier_im.create_dataset('trans', shape=(1004,1004, 3072), dtype='complex')
    transs[:,:,:na] = ali
t2 = time()
t2 - t1
#%%

