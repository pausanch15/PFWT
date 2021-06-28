#Aca trato de adaptar la forma en la que Pablo encuentra los bordes a una funcion que lo haga en general.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import thin, skeletonize, erosion, opening
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import scipy.ndimage as ndi
import networkx as nx
import Func_Splines as spl
import Func_Genera_Fibras as gf
from skimage.color import rgb2gray
from scipy.interpolate import CubicSpline

#Genero imagenes de fibras y encuentro la fibra en cada una
np.random.seed(15)
n = 2
imagenes, fib = gf.crear_im_fibra(n+1,ruido=0.0015, fondo=0.05, salto=10, drift=0)
fibras = spl.encuentra_fibra(imagenes)

#Pruebo encontrar los bordes de esta forma
selems = list()

selems.append(np.array([[0, 1, 0],
                        [0, 1, 0],
                        [0, 0, 0]]))
selems.append(np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]]))
selems = [np.rot90(selems[i], k=j) for i in range(2) for j in range(4)]

endpoints = np.zeros_like(fibras[0], dtype=bool)
for selem in selems:
    endpoints |= ndi.binary_hit_or_miss(fibras[0], selem)

#Grafico
# plt.figure()
# plt.imshow(endpoints.astype(float) + fibras[0].astype(float))
# plt.title('Bordes')
# plt.colorbar()
# plt.show()

#Pasa esto de que el parametro de binarizacion no es suficiente para distinguir la fibra de todo el ruido. Hay que mejorar eso, pero pareciera encontrar bien los bordes.

#Armo la funcion
#Lo que recibe es la imagen de la fibra
#Lo que devuelve es un array del mismo tamaño de la imagen de todos ceros salvo en el lugar de la fibra (donde hay unos) y en los extremos (donde hay dos).
def encuentra_bordes(im_fibra):
    selems = list()
    selems.append(np.array([[0, 1, 0],
                            [0, 1, 0],
                            [0, 0, 0]]))
    selems.append(np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]]))
    selems = [np.rot90(selems[i], k=j) for i in range(2) for j in range(4)]
    endpoints = np.zeros_like(im_fibra, dtype=bool)
    for selem in selems:
        endpoints |= ndi.binary_hit_or_miss(im_fibra, selem)
    return endpoints.astype(float) + im_fibra.astype(float)

#Pruebo la funcion
# for fib in fibras:
    # extremos = encuentra_bordes(fib)
    # plt.figure()
    # plt.imshow(extremos)
    # plt.title('Bordes')
    # plt.colorbar()
    # plt.show()

#Ahora deberíamos hacer esto en cada tramo. Busco tramos.
#Lo que devuelve la funcion es un array del mismo tamaño de la imagen de todos ceros salvo en el lugar de la fibra (donde hay unos) y en los nots (donde hay dos).
def encuentra_ramas(im_fibra):
    #3 branch branching
    selems = list()
    selems.append(np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 0, 0]]))
    selems.append(np.array([[1, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]]))
    selems.append(np.array([[1, 0, 1],
                            [0, 1, 0],
                            [0, 1, 0]]))
    selems.append(np.array([[0, 1, 0],
                            [1, 1, 0],
                            [0, 0, 1]]))
    selems = [np.rot90(selems[i], k=j) for i in range(4) for j in range(4)]
    #4 branch branching
    selems.append(np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]]))
    selems.append(np.array([[1, 0, 1],
                            [0, 1, 0],
                            [1, 0, 1]]))
    branches = np.zeros_like(im_fibra, dtype=bool)
    for selem in selems:
        branches |= ndi.binary_hit_or_miss(im_fibra, selem)
    return branches.astype(float) + im_fibra.astype(float)

#Pruebo la funcion
# for fib in fibras:
    # nots = encuentra_ramas(fib)
    # plt.figure()
    # plt.imshow(nots)
    # plt.title('Nudos')
    # plt.colorbar()
    # plt.show()

#Pruebo una forma de separar las ramas
for fib in fibras:
    extremos = encuentra_bordes(fib)
    nots = encuentra_ramas(fib)
    
    #A la fibra le resto los limites
    #A la hora de pegar los tramos se le vuelven a sumar.
    limites = np.where(nots<2, 0, 1)
    ramas = fib-limites
    
    plt.figure()
    plt.imshow(ramas)
    plt.title('Tramos')
    plt.colorbar()
    plt.show()

#Trato de encontrar bordes de cada rama
for fib in fibras:
    extremos = encuentra_bordes(fib)
    nots = encuentra_ramas(fib)
    limites = np.where(nots<2, 0, 1)
    ramas = fib-limites
    extremos_ramas = encuentra_bordes(fib-limites)
    plt.figure()
    plt.imshow(extremos_ramas)
    plt.title('Extremos de Cada Tramo')
    plt.colorbar()
    plt.show()

#Trato de reemplazar esta forma de encontrar los bordes de cada tramo en ordena_fibra, que parece ser donde trae problema.
#No entiendo que es el ind de la funcion, pero si es la posicion del extremo en el array de x siento que con esto se puede encontrar facil.
#Como input lleva: encuentra_bordes(fibra-np.where(nots<2, 0, 1))
def buscar_bordes(extremos_ramas):  
    xy = np.where(extremos_ramas>1)
    xb = xy[1]
    yb = xy[0]
    return xb, yb

#Pruebo
for fib in fibras:
    extremos = encuentra_bordes(fib)
    nots = encuentra_ramas(fib)    
    limites = np.where(nots<2, 0, 1)
    ramas = fib-limites
    extremos_ramas = encuentra_bordes(fib-limites)
    x_borde, y_borde = buscar_bordes(extremos_ramas)
    plt.figure()
    plt.plot(x_borde, y_borde, 'ro')
    plt.imshow(extremos_ramas)
    plt.title('Extremos de Cada Tramo')
    plt.colorbar()
    plt.show()
