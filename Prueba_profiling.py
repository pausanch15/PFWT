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
import cProfile
from time import time
from sklearn.neighbors import NearestNeighbors
import networkx as nx
#%%
np.random.seed(12)
n = 30
imagenes, fib = gf.crear_im_fibra(n+1,ruido=0.0015,fondo=0.05,salto=10,drift=0)
#%%
t1 = time()
fibras = spl.encuentra_fibra(imagenes,binariza=90)
t2 = time()
t2-t1
#%%
ff = 25
fibra = fibras[ff]
t1 = time()
tramos,bordes = spl.cortar_fibra(fibra,cortar_ruido=False)
t2 = time()
print(t2-t1)
print(len(tramos),bordes)
tramos = spl.ordenar_fibra(tramos)
t3 = time()
print(t3-t2)
curv,spline = spl.pegar_fibra(tramos,bordes,window=21,s=10)
t4 = time()
print(t4-t3)

t_spl = np.linspace(0,1,10000)
xf,yf = splev(t_spl,spline)


plt.figure()
plt.set_cmap('gray')
plt.imshow(imagenes[ff])
#plt.imshow(fibra)
plt.show()
plt.figure()
plt.imshow(fibra)    
plt.plot(xf,yf,'r-')
plt.show()
#%%
#cProfile.run('spl.cortar_fibra(fibra,cortar_ruido=True)',sort='tottime')
cProfile.run('spl.ordenar_fibra(tramos)',sort='tottime')
#cProfile.run('spl.pegar_fibra(tramos,bordes,window=21,s=10)',sort='tottime')
#%%
#De encuentra_fibra lo que mas tiempo parece llevar es 
#built-in method scipy.ndimage._nd_image.correlate
#no se que es ni de donde sale
#
#De cortar_fibra, hacer el convolve2d es lo que lleva más
#tiempo, usar el cortar_ruido=False me ahorra llamarlo 
#2 veces por cada vez que uso la funcion.
#
#De ordenar_fibra, buscar_bordes es lo que lleva más 
# tiempo, vi de 0.44 y 0.93 seg. Depende mucho del largo de los tramos 
#
#Si la fibra no tiene cruces, pegar_fibra no lleva casi +
#tiempo, con nudo tampoco pareciera llevar mucho más tiempo
# Pareciera pasar que a veces tarda mucho esto pero en el cProfile no
#puedo ver porque

#%%
#ff = 26 da problemas (para verlo lo puedo comparar con ff=25)
def main():
    ff = 26
    fibra = fibras[ff]
    tramos,bordes = spl.cortar_fibra(fibra,cortar_ruido=False)
    tramos = spl.ordenar_fibra(tramos)
    curv,spline = spl.pegar_fibra(tramos,bordes,window=21,s=10)

cProfile.run('main()',sort='tottime')
#%%
fibras = spl.encuentra_fibra(imagenes,binariza=90)
splines = []
for ff in range(len(imagenes)):
    print(ff,end=' ')
    fibra = fibras[ff]
    tramos,bordes = spl.cortar_fibra(fibra,cortar_ruido=False)
    tramos = spl.ordenar_fibra(tramos)
    curv,spline = spl.pegar_fibra(tramos,bordes,window=21,s=10)
    splines.append(spline)
#%%
def main2():
    fibras = spl.encuentra_fibra(imagenes,binariza=90)
    splines = []
    for ff in range(len(imagenes)):
#        print(ff,end=' ')
        fibra = fibras[ff]
        tramos,bordes = spl.cortar_fibra(fibra,cortar_ruido=False)
        tramos = spl.ordenar_fibra(tramos)
        curv,spline = spl.pegar_fibra(tramos,bordes,window=21,s=10)
        splines.append(spline)

cProfile.run('main2()',sort='tottime')

#%%
