#Librerías
import h5py 
from tqdm import tqdm
import numpy as np
# import matplotlib.pyplot as plt
# import cv2
from skimage.transform import warp_polar
# plt.ion()

#%%
#Traigo los hdf5 con las alturas, y transformo fourier en 2d cada imagen
#Paso de un array de la forma (t, x, y) a otro que sea (t, kx, ky)
with h5py.File('18-12-2021_fourier_alturas_t.hdf5', 'w') as f:
    fourier_im = f.create_group('fi')
    transs = fourier_im.create_dataset('trans', shape=(3072,1024,1024), dtype='complex')
    with h5py.File('18-12-2021_ftp.hdf5', 'r') as f:
        gdp = f.get('hs')
        alturas = gdp['alts']
        for i, j in zip(range(len(alturas)), tqdm(range(len(alturas)))):
            # transformada = np.abs(np.fft.fft2(alturas[i]))
            transformada = np.fft.fft2(alturas[i])
            transs[i] = transformada
            del(transformada)

#%%
#Traigo los hdf5 con las alturas transformadas y las paso a polares
#Esto ultimo lo guardo en otro hdf5 de la fomra (t, r, thehta)
with h5py.File('18-12-2021_fourier_alturas_polares.hdf5', 'w') as f:
    polares = f.create_group('polares')
    im_pol = polares.create_dataset('imag_pol', shape=(3072,360,725), dtype='float')
    with h5py.File('18-12-2021_fourier_alturas_t.hdf5', 'r') as f:
        gdp = f.get('fi')
        im_trans = gdp['trans']
        for i, j in zip(range(len(im_trans)), tqdm(range(len(im_trans)))):
            im_pol[i] = warp_polar(np.abs(im_trans[i]), center=(512, 512))


