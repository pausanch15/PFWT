#Combino los cambios de ayer durante la llamada con la version orginal del prueba_splines. Trato de hacer estadistica de otra forma.
#Cambio tambien la funcion para que el numero a partir del cual binarizamos sea el que charlamos con Pablo.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import splev, splrep, splprep
from skimage.morphology import thin, skeletonize, remove_small_objects, binary_dilation, dilation
from skimage.util import random_noise
from skimage.filters import gaussian
import Func_Genera_Fibras_2 as gf
#plt.ion()
import Func_Splines as spl
from time import time
#%%
#Generamos n fibras
np.random.seed(12)
t1 = time()
imagenes, splineso = gf.genera_im_dinamica(frames=1,n_fibras=100,drift=50)
t2 = time()
t2-t1, len(imagenes)
#%%
t1 = time()
fibras,bbs = spl.encuentra_fibra(imagenes,binariza=50)
splines = []
for ff in range(len(imagenes)):
#    print(ff,end=' ')
    fibra,bb = fibras[ff], bbs[ff]
    tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=False)
    tramos = spl.ordenar_fibra(tramos)
    curv,spline = spl.pegar_fibra(tramos,bordes,window=21,s=10)
    splines.append(spline)
t2 = time()
t2-t1
#%%
#Probamos solo con una
ff = 59 #93,88,76,60,51,18,8,5,3,42,59

t_spl = np.linspace(0, 1, 10000)
xf, yf = splev(t_spl, splines[ff])
yo, xo = splev(t_spl, splineso[ff])

plt.figure()
plt.set_cmap('gray')
plt.imshow(imagenes[ff])
plt.plot(xf, yf, 'r-')
plt.plot(xo, yo, 'g-')
#plt.plot(xf-xo, 'r-')
#plt.plot(yf-yo, 'g-')
plt.show()
#%%
dx, dy, dxdy = [], [], []
t_spl = np.linspace(0,1,10000)
for i in range(len(imagenes)):
    if i in [3,5,8,42,51,60,88,93]: continue
    xf,yf = splev(t_spl,splines[i])
    yo,xo = splev(t_spl,splineso[i])
    if np.max(np.abs(xf-xo)) > 20 or np.max(np.abs(yf-yo)) > 20:
        xo = xo[::-1]
        yo = yo[::-1]
    if np.max(np.abs(yf-yo)) > 20: print(i)
    dx = dx + list(xf-xo)
    dy = dy + list(yf-yo)
    dxdy = dxdy + list( (xf-xo)+(yf-yo) )
#%%
plt.figure()
plt.hist(dx,bins=50,color='blue',label='x',alpha=0.5)
plt.hist(dy,bins=50,color='red',label='y',alpha=0.5)
plt.legend()
plt.show()
plt.figure()
plt.hist(dxdy,bins=50)
plt.title('(xf-xo)+(yf-yo)')
plt.show()

np.mean(dxdy), np.std(dxdy)
#%%
#Vemos el histograma de la imagen, su media y esa media menos tres desvios (esta cantidad es la que fijaria el parametro de binarizacion en cada caso)
plt.figure()
plt.hist(imagenes[ff].flatten(), bins=100, facecolor='k')
plt.vlines(np.mean(imagenes[ff].flatten()), 0, 570000, color='r', label='Media')
plt.vlines(np.mean(imagenes[ff].flatten())-3*(np.std(imagenes[ff].flatten())), 0, 570000, color='g', label='Media menos Tres Desvíos')
plt.legend()
plt.show()

#Ya cambie la funcion en Func_Splines

#Ahora trato de hacer estadistica.
fibra = fibras[ff]

tramos, bordes = spl.cortar_fibra(fibra, cortar_ruido=True)
tramos = spl.ordenar_fibra(tramos)
curv, spline = spl.pegar_fibra(tramos, bordes, window=21, s=10)

t_spl = np.linspace(0, 1, 10000)

#xf y yf son as coordenadas de la fibra
xf, yf = splev(t_spl, spline)

#Vemos la fibra que recuperamos
plt.figure()
plt.imshow(fibra)    
plt.plot(xf, yf, 'r-')
plt.show()

#xo y yo son las coordenadas de la fibra original
yo, xo = splev(t_spl, fib[ff])

#Comparo la fibra original con la que analizamos
plt.figure()
plt.plot(xo, yo, 'g-', label='Fibra Original') 
plt.plot(xf, yf, 'r-', label='Fibra Obtenida')
plt.legend()
plt.show()

plt.figure()
plt.plot(xf-xo, label='Diferencia en x')
plt.plot(yf-yo, label='Diferencia en y')
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.plot(np.abs(xf-xo), label='Diferencia en x')
plt.plot(np.abs(yf-yo), label='Diferencia en y')
plt.grid()
plt.legend()
plt.show()

#Hago el histograma para la diferencia en x y en y. Cuantos puntos difieren en cada valor?
plt.figure()
plt.hist(xf-xo, bins=30, facecolor='blue', label='x', alpha=0.5, edgecolor='k')
plt.hist(yf-yo, bins=30, facecolor='red', label='y', alpha=0.5, edgecolor='k')
plt.legend()
plt.show()

plt.figure()
plt.hist(np.abs(xf-xo), bins=30, facecolor='blue', label='x', alpha=0.5, edgecolor='k')
plt.hist(np.abs(yf-yo), bins=30, facecolor='red', label='y', alpha=0.5, edgecolor='k')
plt.legend()
plt.show()
