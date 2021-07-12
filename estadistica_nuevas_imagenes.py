#Acá pruebo hacer estadística con las imgágenes generadas con los códigos que pasó Pablo y las funciones mejoradas.
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from scipy.interpolate import splev, splrep, splprep
from skimage.morphology import thin, skeletonize, remove_small_objects, binary_dilation, dilation
from skimage.util import random_noise
from skimage.filters import gaussian
import Func_Genera_Fibras_2 as gf
#plt.ion()
import Func_Splines as spl
from time import time
#%%
##Genero la fibra inicial, la final y la dinamica entre ellas
#np.random.seed(12)
#fib_i = gf.generar_fibra(N=100, L=1, alpha=0.1, Nt=8)
#fib_int = gf.generar_fibra(N=100, L=1, alpha=0.1, Nt=8)
#fib_f = gf.generar_fibra(N=100, L=1, alpha=0.1, Nt=8)
##dinam = gf.generar_dinamica_entre_dos_instancias(fib_i, fib_f, Nsteps=10)
#dinam = gf.generar_dinamica([fib_i,fib_int,fib_f], Nsteps=10)
#
##Genero las imágenes con ruido de estas fibras
#imagenes = [gf.genera_im_fibra(dinam[:, i]) for i in range(np.shape(dinam)[-1])]
#%%
np.random.seed(12)
t1 = time()
imagenes, splineso = gf.genera_im_dinamica(n_fibras=3,drift=50)
t2 = time()
t2-t1
#%%
#Veo algunas imágenes, para ver cómo son
#for im in imagenes[::5]:
#    plt.figure()
#    plt.imshow(im, cmap='gray')
#    plt.colorbar()
#    plt.show()
#%%
#Trato de encontrar las fibras en cada imagen
t1 = time()
fibras, bbs = spl.encuentra_fibra(imagenes,binariza=50)
t2 = time()
t2-t1

#Veo cómo encuentra las fibras
#for im in fibras[4:7]:
#    plt.figure()
#    plt.imshow(im, cmap='gray')
#    plt.colorbar()
#    plt.show()
#%%
#Hago lo mismo que hacíamos antes para hacer estadística
#Elijo una de las fibras que encontramos en las imágenes, cualquiera.
ff = 12
fibra, bb = fibras[ff], bbs[ff]

#Interpolo la fibra encontrada
t1 = time()
tramos, bordes = spl.cortar_fibra_rap(fibra, bb, cortar_ruido=False)
tramos = spl.ordenar_fibra(tramos)
curv, spline = spl.pegar_fibra(tramos, bordes, window=21, s=10)
t2 = time()
print(t2-t1)

t_spl = np.linspace(0, 1, 10000)

#xf y yf son as coordenadas de la fibra
xf, yf = splev(t_spl, spline)
yo, xo = splev(t_spl, splineso[ff])
#Vemos la fibra que recuperamos
plt.figure()
#plt.imshow(imagenes[ff],cmap='gray')    
#plt.imshow(fibra,cmap='gray')    
#plt.plot(xf, yf, 'r-')
#plt.plot(xo, yo, 'g-')
plt.plot(xf-xo, 'r-')
plt.plot(yf-yo, 'g-')
plt.show()
#%%
t1 = time()
fibras,bbs = spl.encuentra_fibra(imagenes,binariza=50)
splines = []
for ff in range(len(imagenes)):
#    print(ff,end=' ')
    fibra,bb = fibras[ff], bbs[ff]
    tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=False)
    tramos = spl.ordenar_fibra(tramos)
    curv,spline = spl.pegar_fibra(tramos,bordes,window=21,s=10)
    splines.append(spline)
t2 = time()
t2-t1
#%%
dx, dy, dxdy = [], [], []
t_spl = np.linspace(0,1,10000)
for i in range(len(imagenes)):
    if i == 13: continue
    xf,yf = splev(t_spl,splines[i])
    yo,xo = splev(t_spl,splineso[i])
    if np.max(np.abs(xf-xo)) > 20:
        xo = xo[::-1]
        yo = yo[::-1]
    if np.max(np.abs(yf-yo)) > 4: print(i)
    dx = dx + list(xf-xo)
    dy = dy + list(yf-yo)
    dxdy = dxdy + list( (xf-xo)+(yf-yo) )
    
plt.figure()
plt.hist(dx,bins=50,color='blue',label='x',alpha=0.5)
plt.hist(dy,bins=50,color='red',label='y',alpha=0.5)
plt.legend()
plt.show()
plt.figure()
plt.hist(dxdy,bins=50)
plt.title('(xf-xo)+(yf-yo)')
plt.show()

np.mean(dxdy), np.std(dxdy)
#%%
#xo y yo son las coordenadas de la fibra original. Es la fibra que está en dinam[:, ff]
y, x = np.real(dinam[:, ff]), np.imag(dinam[:, ff])
spl, u = splprep([x, y], s=0)
xo, yo = splev(t_spl, spl)

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
