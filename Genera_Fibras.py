#Acá están las funciones dedicadas a generar fibras dinámicas par testear el código.
from math import * #Creo que no lo uso, lo usé par probar cosas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from scipy.interpolate import splev, splrep, splprep
from skimage.util import random_noise
from skimage.filters import gaussian

def arma_fibra_dinamica(x, y, frames, salto):
# la cambie para que empiece del (x,y) y los frames siguientes pueda controlar cuanto saltan con salto
    xs,ys = np.array(x),np.array(y)
    N_puntos = len(x)
    fibra = []
    for i in range(frames):
        #global x,y
        xs,ys = xs+(salto*np.random.random(N_puntos)-salto/2), ys+(salto*np.random.random(N_puntos)-salto/2)
        spl, u = splprep([xs, ys], s=0)
        t_spl = np.linspace(0, 1, 1000)
        frame = splev(t_spl, spl)
        fibra.append(frame)
    return fibra #[[fibra1_x, fibra1_y], [fibra2_x, fibra2_y], ...]

def arc_length(x, y):
    npts = len(x)
    arc = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
    for k in range(1, npts):
        arc = arc + np.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2)
    return arc

def fija_largo(fibra): #Solo sirve si las fibras son más largas que la primera
    nueva_fibra = fibra.copy() #Para no perder la original
    largo = arc_length(*nueva_fibra[0])
    for frame in nueva_fibra: #saque lo de verificar si el largo era igual para que sea un poco mas rapido (total no hacias nada con eso más que los prints)
        while arc_length(*frame) > largo:
            frame[0] = frame[0][:-1]
            frame[1] = frame[1][:-1]
        # print(arc_length(*frame), end=' ') #Dejé los prints para que se vea por cuanto le pifia
#         print() #Dejé los prints para que se vea por cuanto le pifia
#         print() #Dejé los prints para que se vea por cuanto le pifia
    return nueva_fibra


def crear_im_fibra(frames,n=4,sigma=1,ruido=0.003,fondo=0.05,salto=20):
# empiezo en algun punto random (que no sea muy cera de los bordes) de mi imagen 1000x1000
    x,y = [],[]
    x.append(np.random.rand()*890+50) 
    y.append(np.random.rand()*890+50)
    for i in range(n-1):
# busco quedarme cerca del punto anterior
        x.append(x[0]+np.random.random()*200-100)
        y.append(y[0]+np.random.random()*200-100)
    fibra = fija_largo(arma_fibra_dinamica(x,y, frames,salto))
    imagenes = []
    for frame in fibra:
        im_fibra, xedges, yedges = np.histogram2d(*frame, 1000, [[0,1000],[0,1000]])
        im_fibra = im_fibra==0
        im_fibra = random_noise(im_fibra, mode='s&p', amount=ruido)
        ff = np.linspace(0,999,1000)
        fx,fy = np.meshgrid(ff,ff)
        im_fibra = (im_fibra + fx*fy / 1000**2 * fondo) / 2  #gaussian(np.ones_like(im_fibra),sigma=1)
        im_fibra = gaussian(im_fibra, sigma=sigma)
        im_fibra = np.array(255*im_fibra, dtype = 'uint8')
        imagenes.append(im_fibra)
    return imagenes
#%%