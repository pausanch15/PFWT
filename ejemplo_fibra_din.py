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
#%%
np.random.seed(12)
n = 20
imagenes,fibra = gf.crear_im_fibra2(n+1,fondo=0.05,salto=10,ruido=0.0015)
#%%
im = 0

t_spl = np.linspace(0,1,10000)
x,y = splev(t_spl,fibra[im])

plt.figure()
plt.imshow(imagenes[im])
#plt.gca().invert_yaxis()
plt.set_cmap('gray')
plt.plot(y,x,'r-')
plt.show()

#%%
import imageio

np.random.seed(12)
n = 40
imagenes = gf.crear_im_fibra(n+1,fondo=0.05,salto=5)

imageio.mimsave('C:\\Users\\tomfe\\Documents\\TOMAS\\Facultad\\Laboratorio 6\\fibra.gif',imagenes,fps=12)
#%%