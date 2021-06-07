#Tomo el código de prueba_splines (el que hace todo y tiene la función que encuentra la fibra en la imagen), trato de quedarme con las posiciones de los puntos de las fibras y guardarlos con h5py.
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

#Armamos imagenes de la fibra dinamica y encontramos la fibra en ellas.
np.random.seed(3)
n = 5
imas = gf.crear_im_fibra(n+1,fondo=0.05)
im_nombres = ps.guarda_imagenes(imas, 'test')

imagenes = []
for im in im_nombres:
    imagenes.append(Image.open(im))

fibras = ps.encuentra_fibra(imagenes)

for fibra in fibras:
    tramos,bordes = spl.cortar_fibra(fibra)
    for tr in range(len(tramos)):
        x,y = tramos[tr][:,0], tramos[tr][:,1]
        plt.plot(x, y, 'o', color='grey')
    tramos = spl.ordenar_fibra(tramos)
    t, curv, xf, yf = spl.pegar_fibra(tramos,bordes)
    
    plt.plot(xf, yf, 'r-')
    
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    plt.show()

#Trato de guardar los x e y de cada fibra en archivos de hf5.
#Estos van de prueba.

f = h5py.File('Datos.hdf5', 'w')
grp = f.create_group("Fibra_0")
subgrp_0 = grp.create_group("x")
subgrp_1 = grp.create_group("y")
print(f'Grupo: {grp.name}')
print(f'Primer Subgrupo: {subgrp_0.name}')
print(f'Segundo Subgrupo: {subgrp_1.name}')
f.close()

#Ahora trato de usarlos
f = h5py.File('Datos.hdf5', 'w')
grp = f.create_group("Fibra_0")

for fibra, nom in zip(fibras, im_nombres):
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
    
