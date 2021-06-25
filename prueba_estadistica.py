#Combino los cambios de ayer durante la llamada con la version orginal del prueba_splines. Trato de hacer estadistica de otra forma.
#Cambio tambien la funcion para que el numero a partir del cual binarizamos sea el que charlamos con Pablo.
import Func_Splines as spl
import Func_Genera_Fibras as gf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import thin, skeletonize, remove_small_objects, binary_dilation, dilation
from scipy.interpolate import CubicSpline, splev, splrep, splprep
from scipy.signal import convolve2d, savgol_filter
from itertools import permutations
import imageio

#Generamos n fibras
np.random.seed(12)
n = 10
imagenes, fib = gf.crear_im_fibra(n+1, ruido=0.0015, fondo=0.05, salto=50, drift=0)
fibras = spl.encuentra_fibra(imagenes)

#Probamos solo con una
ff = 5

plt.figure()
plt.set_cmap('gray')
plt.imshow(imagenes[ff])
#plt.imshow(dilation(255-imagenes[ff]))
plt.show()

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
