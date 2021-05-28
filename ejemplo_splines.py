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

np.random.seed(12)
n = 40
imagenes = gf.crear_im_fibra(n+1,fondo=0.05,salto=5)
#%%
imt = imagenes[5] < 108
iml = label(imt)
prop_reg = regionprops(iml)
for i in range(len(prop_reg)):
    if prop_reg[i].area > 100:
        fibra = iml==i+1
fibra = skeletonize(fibra)

plt.figure()
plt.set_cmap('jet')
#plt.imshow(imt)
plt.imshow(fibra)
plt.show()

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
#plt.tick_params(left = False, right = False , labelleft = False ,
#                labelbottom = False, bottom = False)
plt.show()
#%%

np.random.seed(12)
n = 1
imagenes = gf.crear_im_fibra(n+1,fondo=0.05,salto=5)

lista_im = []
for frames in range(len(imagenes)):

    imt = imagenes[0] < 115
    iml = label(imt)
    prop_reg = regionprops(iml)
    for i in range(len(prop_reg)):
        if prop_reg[i].area > 100:
            fibra = iml==i+1
    fibra = skeletonize(fibra)

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
    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
    plt.savefig('imageni.png',layout='tight')
    lista_im.append(imageio.imread('imageni.png'))
    

