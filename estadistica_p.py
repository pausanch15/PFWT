#Pruebo hacer estadística con muchas imágenes.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import splev, splrep, splprep
from skimage.morphology import thin, skeletonize, remove_small_objects, binary_dilation, dilation
from skimage.util import random_noise
from skimage.filters import gaussian
import Func_Genera_Fibras_2 as gf
import Func_Splines as spl
from time import time
plt.ion()

#Genero las imágenes con ruido de estas fibras. Me fijo cuánto tarda en hacer esto. Genero n imágenes entre cada imagen extremo. Uso e extremos (imágenes intermedias).
n = 1000
e = 3
np.random.seed(12)
ti = time()
imagenes, splineso = gf.genera_im_dinamica(frames=n, n_fibras=e, drift=50)
tf = time()
print(f'Tarda {tf-ti} segundos en generar {len(imagenes)} imágenes.')

#Hago esto para todas las imágenes generadas.
#Interpolo todas las fibras. Me fijo cuánto tarda
ti = time()
fibras, bbs = spl.encuentra_fibra(imagenes,binariza=50)
print(f'Tarda {tf-ti} segundos en encontrar las fibras de las {len(imagenes)} imágenes.')
splines = []
for ff in range(len(fibras)):
#   print(ff,end=' ')
    fibra,bb = fibras[ff], bbs[ff]
    tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=False)
    tramos = spl.ordenar_fibra(tramos)
    curv,spline = spl.pegar_fibra(tramos,bordes,window=21,s=10)
    splines.append(spline)
tf = time()
print(f'Tarda {tf-ti} segundos en interpolar todas las fibras de las imágenes.')

#Hago el histograma para todas
dx, dy, dxdy = [], [], []
t_spl = np.linspace(0,1,10000)
for i in range(len(fibras)):
    if i == 13: continue
    xf,yf = splev(t_spl,splines[i])
    yo,xo = splev(t_spl,splineso[i])
    if np.max(np.abs(xf-xo)) > 20:
        xo = xo[::-1]
        yo = yo[::-1]
    if np.max(np.abs(yf-yo)) > 4: print(i)
    dx = dx + list(xf-xo)
    dy = dy + list(yf-yo)
    dxdy = dxdy + list((xf-xo)+(yf-yo))
    
# plt.figure()
# plt.hist(dx, bins='auto', color='blue', label='x', alpha=0.5, density=True, stacked=True)
# plt.hist(dy, bins='auto', color='red', label='y', alpha=0.5, density=True, stacked=True)
# plt.legend()
# plt.show()
# plt.figure()
# plt.hist(dxdy,bins='auto')
# plt.title('(xf-xo)+(yf-yo)')
# plt.show()

