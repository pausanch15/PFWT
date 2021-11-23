import numpy as np
from scipy import signal
import scipy.fft as scf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# The following line requires "unwrap" to be installed (this is a very robust 2D unwrapper)
# the command to install it from pip is simply: "pip install unwrap" 
#from unwrap import unwrap
from skimage.restoration import unwrap_phase as unwrap
from skimage.io import imread
#%%
# traigo imagen
imas = []
for i in range(1,4):
    ni = '{:04d}'.format(i)
    # ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\Escala\ID_0_C1S000200'+ni+'.tif')
    ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0002/ID_0_C1S000200'+ni+'.tif')
    ima = np.array(ima,dtype='float64')
    imas.append(ima)
#%%
# binariro imagen
i = 2
plt.figure()
plt.imshow(imas[i],cmap='gray')
plt.colorbar()
plt.show()

im = imas[i]
cor = im<200
corte = im * cor
cor = corte>50
plt.figure()
plt.imshow(cor,cmap='gray')
#plt.colorbar()
plt.show()
#%%
# hace basicamente todo, no pregunten
x = np.linspace(750,849,100)
pix = []
for n in range(750,850):
    lin = cor[n,:]
    x0 = np.linspace(0,1023,1024)
    he = np.heaviside(x0-510,0)
    lin = lin*he
    x1 = np.ones_like(lin[550:880])
    lin[550:880] = x1
    pix.append( np.sum(lin) )
    
plt.figure()
plt.plot(x,pix)
plt.grid()
plt.show()

y = np.linspace(690,749,60)
pix = []
for n in range(690,750):
    lin = cor[:,n]
    x0 = np.linspace(0,1023,1024)
    he = np.heaviside(x0-604,0)
    lin = lin*he
    x1 = np.ones_like(lin[630:970])
    lin[630:970] = x1
    pix.append( np.sum(lin) )
    
plt.figure()
plt.plot(y,pix)
#plt.plot(lin)
plt.grid()
plt.show()

#teniamos que el largo del coso es (8.20 +- 0.05) cm
#viendo los grafico para x e y del radio, diria que es (390 +- 2) pixeles

#propagando errores me queda que cada pixel equivale a (0.02106 +- 0.00017) cm
#entonces la imagen completa serian (21.56 +- 0.17) cm                                                       
