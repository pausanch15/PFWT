#Pruebo automatizar como nos quedamos con la region de la imagen que tiene la fibra, yy como binarizar
#remove_small_objects parece que puede ayudar a quedarse solo con la region de la fibra
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from scipy.interpolate import splev, splrep, splprep, CubicSpline
from skimage.util import random_noise
from skimage.filters import gaussian
from ipywidgets import interact
import Genera_Fibras as gf
import Splines as spl
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import thin, skeletonize, remove_small_objects
from scipy.signal import convolve2d, savgol_filter
from itertools import permutations

#Funciones útiles
def print_im(frame):
    # global imagenes
    plt.figure()
    plt.imshow(frame)
    plt.set_cmap('gray')
    plt.colorbar()
    plt.show()

def guarda_imagenes(imagenes, nombre_tira):
    num = [str(i) for i in range(len(imagenes))]
    nombres = [f'{nombre_tira}_{nu}.png' for nu in num]
    for im, nom in zip(imagenes, nombres):
        # print_im(im)
        ima = Image.fromarray(im)
        ima.save(nom, 'png')
        return nombres

#Primero genero algunas imagenes, las guardo y las traigo
np.random.seed(3)
n = 3
imas = gf.crear_im_fibra(n+1,fondo=0.05)
im_nombres = guarda_imagenes(imas, 'test')

imagenes = []
for im in im_nombres:
    imagenes.append(Image.open(im))

# for im in imagenes:
    # im = np.asarray(im)
    # im = im<97 #Esto va a tener que ser un input de la funcion
    # fibra = thin(remove_small_objects(im, connectivity=4))
    # print_im(fibra)

#Armo la funcion que agarra la imagen, la convierte en un array, la binariza, se queda solo con la fibra y le hace el thin.
#No se me ocurre como hacer que el parametro para binarizar no sea un input. Como default le dejo 97 porque pareciera que funciona bien con las imagenes 
#Estoy suponiendo que si eventualmente tenemos imagenes en las que se ven turbinas o algo asi van a estar en los bordes de la imagen, y en ese caso solo deberiamos cortar la misma cantidad de margen en cada caso, lo que se puede agregar facilmente.

def encuentra_fibra(imagenes, binariza=97):
    fibras = []
    for im in imagenes:
        im = np.asarray(im)
        im = im<binariza #Esto va a tener que ser un input de la funcion
        li = label(im)
        fibra = thin(remove_small_objects(im, connectivity=4))
        fibras.append(fibra)
    return fibras

#Pruebo el codigo que tenemos en imagenes generadas
fibras = encuentra_fibra(imagenes)

for fibra in fibras:
    tramos,bordes = spl.cortar_fibra(fibra)
    for tr in range(len(tramos)):
        x,y = tramos[tr][:,0],tramos[tr][:,1]
        plt.plot(x,y,'o',color='grey')
    tramos = spl.ordenar_fibra(tramos)
    t,curv,xf,yf = spl.pegar_fibra(tramos,bordes)
    
    plt.plot(xf,yf,'r-')
    
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    plt.show()
