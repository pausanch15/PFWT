#Ejemplo para la presentacion de como pasamos de la imagen original de la fibra a la fibra interpolada.
import Splines as spl
import Genera_Fibras as gf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import thin, skeletonize
from scipy.interpolate import CubicSpline, splev, splrep, splprep
from scipy.signal import convolve2d, savgol_filter
from itertools import permutations
import imageio
#%%
image = Image.open('imagenprueba.png') #cargo la imagen
ima = np.asarray(image) #la convierto en numpy array

im = ima[:,:,0] #eligo quedarme con el color rojo
imc = im[:, 100:800] #me quedo entre las lineas 100 y 800 (quito las turbinas)
imt = imc<200 #pongo el threshold en 200 para binarizar la imagen

li = label(imt) # etiqueto cada región contigua de pixeles llenos
fibra = li==6  #me quedo con la fibra (vimos graficamente que la fibra tenia la label 6)

# fibra = thin(fibra)
fibra = skeletonize(fibra) 

plt.figure()
plt.imshow(fibra)
plt.show()
#%%
plt.figure()
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

#%%
#Veo con una fibra generada
# Buenas: 12, 15
np.random.seed(12)
n = 40
imagenes, fib = gf.crear_im_fibra(n+1,ruido=0.0015,fondo=0.05,salto=5,drift=[0,0])
#%%
imageio.mimsave('C:\\Users\\tomfe\\Documents\\TOMAS\\Facultad\\Laboratorio 6\\Github\\probando.gif',imagenes,fps=12)
#%%
#con seed = 12 en frame 24, 25 o 26 (donde empieza el cruce), preguntar si importa que no cubra toda fibra.
# Mismo problema con seed 15 frame 10, 11 o 18
ff = 25  
imt = imagenes[ff] < 113
iml = label(imt)
prop_reg = regionprops(iml)
for i in range(len(prop_reg)):
    if prop_reg[i].area > 100:
        fibra_g = iml==i+1
fibra = thin(fibra_g)

#plt.figure()
#plt.set_cmap('gray')
#plt.imshow(imagenes[ff]) #,extent=[500,1000,0,350])
#plt.show()
#plt.figure()
#plt.xlim(left=500,rigt=1000)
#plt.imshow(imt)
#plt.imshow(fibra)
#plt.show()
#
plt.figure()
tramos,bordes = spl.cortar_fibra2(fibra)
#
#for tr in range(len(tramos)):
#    x,y = tramos[tr][:,0],tramos[tr][:,1]
#    plt.plot(x,y,'o',color='grey')
##
tramos = spl.ordenar_fibra(tramos)
t,curv,xf,yf = spl.pegar_fibra(tramos,bordes)

fibff = fib[ff] 
plt.plot(fibff[1],fibff[0],'r-')
plt.plot(xf,yf,'g-',alpha=0.75)
plt.show()
plt.show()
plt.plot(fibff[1])
plt.plot(xf[::-1])
plt.plot(fibff[0],'r-')
plt.plot(yf[::-1],'g-')
#plt.xlim(500,1000)
#plt.ylim(0,350)
#plt.gca().invert_yaxis()
#plt.grid(True)
##plt.tick_params(left = False, right = False , labelleft = False ,
##                labelbottom = False, bottom = False)
plt.show()
#%%
xy_g = np.where(fibra_g == True)
yfg, xfg = xy_g[0],xy_g[1]

errores = []
for i in range(len(xfg)):
    dists = np.sqrt((xf - xfg[i])**2 + (yf - yfg[i])**2)
    mdist = np.min(dists)
    errores.append(mdist)
    
print( np.mean(errores))
#%%
lista_im = []
for frames in range(len(imagenes)):
    print(frames)
    imt = imagenes[frames] < 113
    iml = label(imt)
    prop_reg = regionprops(iml)
    for i in range(len(prop_reg)):
        if prop_reg[i].area > 100:
            fibra = iml==i+1
    fibra = skeletonize(fibra)

    tramos,bordes = spl.cortar_fibra2(fibra)
    
#    for tr in range(len(tramos)):
#        x,y = tramos[tr][:,0],tramos[tr][:,1]
#        plt.plot(x,y,'o',color='grey')
    
    tramos = spl.ordenar_fibra(tramos)
    t,curv,xf,yf = spl.pegar_fibra(tramos,bordes)
    plt.ioff()
    plt.figure()
    plt.imshow(imagenes[frames])
    plt.plot(xf,yf,'r-')
    plt.set_cmap('gray')
#    plt.xlim(100,300)
#    plt.ylim(650,825)
#    plt.gca().invert_yaxis()
#    plt.grid(True)
#    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    plt.savefig('imageni.png',bbox_inches='tight')
    lista_im.append(imageio.imread('imageni.png'))
#%%
imageio.mimsave('C:\\Users\\tomfe\\Documents\\TOMAS\\Facultad\\Laboratorio 6\\fibrayspline2.gif',lista_im,fps=12)

