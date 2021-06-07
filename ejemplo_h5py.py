#Tomo las cosas que sirvieron de prueba_guardar_datos y recorro un poco el archivo de h5py que crea.
#https://docs.h5py.org/en/stable/
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from scipy.interpolate import splev, splrep, splprep, CubicSpline
from skimage.util import random_noise
from skimage.filters import gaussian
import Genera_Fibras as gf
import Splines as spl
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import thin, skeletonize, remove_small_objects
from scipy.signal import convolve2d, savgol_filter
from itertools import permutations
import prueba_splines as ps
import h5py

#Armamos imagenes de la fibra dinamica y encontramos la fibra en ellas. Depsues guardamos los x e y de cada frame con h5py.
np.random.seed(3)
n = 5
imas = gf.crear_im_fibra(n+1,fondo=0.05)
im_nombres = ps.guarda_imagenes(imas, 'test')

imagenes = []
for im in im_nombres:
    imagenes.append(Image.open(im))

fibras = ps.encuentra_fibra(imagenes)

f = h5py.File('Datos.hdf5', 'w')
grp = f.create_group("Fibra_0")

for fibra, nom in zip(fibras, range(len(fibras))):
    nom = str(nom)
    tramos,bordes = spl.cortar_fibra(fibra)
    for tr in range(len(tramos)):
        x,y = tramos[tr][:,0], tramos[tr][:,1]
    tramos = spl.ordenar_fibra(tramos)
    t, curv, xf, yf = spl.pegar_fibra(tramos,bordes)

    #Trato de guardar x e y de la fibra de cada frame
    subgrp = grp.create_group(nom)
    subgrp_x = subgrp.create_group("x")
    subgrp_x.create_dataset("x", data=xf)
    subgrp_y = subgrp.create_group("y")
    subgrp_y.create_dataset("y", data=yf)
    
f.close()

#Recorremos un poco Datos.hdf5. Para eso lo abrimos en modo lectura.
f = h5py.File('Datos.hdf5', 'r')

#Primero vemos que tiene un único grupo en este caso: Fibra_0.
print(f'El group en Datos.hdf5 es {list(f.keys())[0]}')

#Sabemos por lo que hicimos antes que tenemos los 6 subgrupos, uno por cada frame (fibra), dentro de Fibra_0. Los nombres de los subgrupos son los que pusimos: 0, 1, 2, 3, 4, 5.
#Entramos al subgrupo 0 y vemos que tiene.
g0 = f.get('Fibra_0/0') #Este es el subgrupo correspondiente al primer frame
print(f'Los items que hay en grupo correspondiente al prmer frame son son {list(g0.items())[0][0]} e {list(g0.items())[1][0]}')

#Y faltaría como ver cada dataframe dontro de g0, pero ya no tengo ganas hoy. Queda para hacerse estos días.

#Finalmente cerramos el archivo.
f.close()
