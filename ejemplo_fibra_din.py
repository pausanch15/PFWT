from math import * #Creo que no lo uso, lo usé par probar cosas
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from scipy.interpolate import splev, splrep, splprep
from skimage.util import random_noise
from skimage.filters import gaussian
from ipywidgets import interact
import Genera_Fibras as gf

np.random.seed(3)
n = 50
imagenes = gf.crear_im_fibra(n+1,fondo=0.05)
def print_im(frame):
    global imagenes
    plt.figure()
    plt.imshow(imagenes[frame])
    plt.set_cmap('gray')
    plt.colorbar()
    plt.show()
    
interact(print_im,frame=(0,10,1))
plt.figure(figsize=(7,7))
plt.imshow(imagenes[-1])
plt.set_cmap('gray')
plt.colorbar()
plt.show()
