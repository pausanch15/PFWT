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
n = 20
imagenes, fib = gf.crear_im_fibra(n+1, ruido=0.0015, fondo=0.05, salto=50, drift=0)
fibras = spl.encuentra_fibra(imagenes)

#Probamos solo con una
ff = 19

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
plt.show()

#Ya cambie la funcion en Func_Splines
