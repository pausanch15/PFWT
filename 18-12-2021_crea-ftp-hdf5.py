#Archivo que crea los hdf5 de las alturas
import numpy as np
import Func_Splines as spl
import Continuacion_Fourier as cf
from skimage.io import imread
from scipy.interpolate import splev
import h5py 
from tqdm import tqdm

#%%
#Traigo imagenes
gris = np.zeros((1024, 1024))
for i in range(1, 16):
    ni = '{:04d}'.format(i)
#    ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0003\ID_0_C1S000300'+ni+'.tif')
    ima = np.array(ima, dtype='float')
    gris += ima
gris = gris/15

ref = np.zeros((1024, 1024))
for i in range(1, 14):
    ni = '{:04d}'.format(i)
#    ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0002/ID_0_C1S000300'+ni+'.tif')
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0002\ID_0_C1S000300'+ni+'.tif')
    ima = np.array(ima, dtype='float')
    ref += ima
ref = ref/13

#%%
#Para pasar de diferencia de fase a altura, necesitamos la longitud de onda del patrón de rayas
from scipy.signal import find_peaks
asd = ref - gris
xp = np.arange(0,1024,1)
x = 0.02086 * xp
n = 200
lin = asd[n,:] - np.mean(asd[n,:])

a, _ = find_peaks(lin,distance=10) #hay 82 picos, o sea, 81 longitudes de onda
d = (1014-9) / 81 * 0.02086
w = 2*np.pi/d
rec = np.cos(w*x + 1.5) * 75

#%%
#Saco altura para las imagenes que tienen la fibra
#Hago ftp y guardo en hf5 las alturas
thx,thy, ns = 0.25, 45, 0.75 #0.5, 80, 0.5

#Defino los valores de L y D (los medimos en el labo)
w = 2*np.pi/d
L, D = 79.6, 20.3

#with h5py.File('18-12-2021_ftp.hdf5', 'w') as f:
with h5py.File('alturas.hdf5', 'w') as f: #le cambio el nombre para mi    
    h_im = f.create_group('hs')
    alt = h_im.create_dataset('alts', shape=(3072,1024,1024))
    for num, i in zip(range(1, 3073), tqdm(range(1, 3073))):
        ni = '{:04d}'.format(num)
#        ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0001/ID_0_C1S000100'+ni+'.tif')
        ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
        im = np.array(ima,dtype='float')
        dph, ft, gf = cf.dphase_2d(im,ref-gris,thx,thy,ns,inde=9)
        altura = (L*dph) / (dph - w*D)
        alt[num-1] = altura - np.mean(altura)
        del(ni, ima, im, altura)


#%%

