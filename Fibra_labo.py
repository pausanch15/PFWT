import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from scipy.interpolate import splev, splrep, splprep
from skimage.morphology import thin, skeletonize, remove_small_objects, binary_dilation, dilation
from skimage.measure import label, regionprops
#from skimage.util import random_noise
#from skimage.filters import gaussian
from skimage.io import imread
import Func_Genera_Fibras_2 as gf
import Func_Splines as spl
from time import time
from scipy import interpolate
import scipy.stats as sps
#from itertools import permutations
import h5py 
plt.ion()
#%%
im_p = np.zeros((1024,1024))
for i in range(1,301):
    ni = '{:04d}'.format(i)
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\ID_0_C1S0003\ID_0_C1S000300'+ni+'.tif')
    im_p += ima
im_p = im_p/300
#%%
plt.figure()
plt.imshow(im_p,cmap='gray')
plt.show()
#%%
imags = []
t1 = time()
for i in range(1,500):
    ni = '{:04d}'.format(i)
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\ID_0_C1S0004\ID_0_C1S000400'+ni+'.tif')
    imags.append(ima)
t2 = time()
print(t2-t1)
#%%
ff = 498
plt.figure()
plt.imshow(imags[ff]-im_p,cmap='gray_r')
plt.show()
bina = np.mean(imags[ff]-im_p) - 63
print(bina, np.mean(imags[ff]-im_p), np.std(imags[ff]-im_p))
#bina = -70
plt.figure()
plt.imshow((imags[ff]-im_p)<bina,cmap='gray_r')
plt.show()             
#%%
fib, splines = [],[]
mcu = []
t1 = time()
for i in range(499):
#    print(i, end=' ')
    bina = np.mean(imags[i]-im_p) - 2.2 * np.std(imags[i]-im_p) #63
    if i in [26,29,30,361]:
        bina = -50
    if i in [31,33,35,37,41,50,54,99]:
        bina = -67
    if i == 361:
        bina = -75
    fibrass,bbs = spl.encuentra_fibra([imags[i]-im_p],binariza=bina,connec=0,eccen=0.998)
    try:
        fibra,bb = fibrass[0], bbs[0]
        fib.append(fibra)    
        tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=True)
        tramos = spl.ordenar_fibra(tramos)
        curv,spline = spl.pegar_fibra(tramos,bordes,window=17,s=10)
        splines.append(spline)
        mcu.append(np.max(curv))
    except: 
        print('no encontró',i)
        fib.append('Nan')
        splines.append('Nan')
        mcu.append('Nan')
t2 = time()
print(t2-t1)
#%%
#for i in range(24,30):
#    plt.figure()
#    plt.imshow(fib[i])
#    plt.show()

for i in range(len(mcu)):
    if mcu[i] == 'Nan': continue
    if mcu[i] > 1: 
        print(mcu[i],i)
#%% 
#8.830121891284316 10
#329.0416777931968 14
#67.25263064573389 210
ff = 210
t_spl = np.linspace(0, 1, 10000)
xf, yf = splev(t_spl, splines[ff])

plt.figure()
plt.set_cmap('gray_r')
plt.imshow(imags[ff]-im_p)
plt.imshow(fib[ff])
plt.plot(xf, yf, 'r-')
plt.show()

#%%
#revisar 11
ff = 428 #361, 427
ni = '{:04d}'.format(ff)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\ID_0_C1S0004\ID_0_C1S000400'+ni+'.tif')
bina = np.mean(ima-im_p) - 2.2 * np.std(ima-im_p) #63
#bina = -75
print(bina, np.mean(ima-im_p), np.std(ima-im_p))
im = (ima-im_p)<bina
     
fibra = remove_small_objects(im, connectivity=1)
li = label(fibra)
print('Area\t BBox_area\t Eccentricity\t Euler_number\t Area_coef\t i')
prop = regionprops(li)
for i in range(np.max(li)):
    ar, ba, ec = prop[i].area, prop[i].bbox_area, prop[i].eccentricity
    en, fa, bar = prop[i].euler_number, prop[i].filled_area, ba/ar
    print('{:<8d} {:<15d} {:<15f} {:<15d} {:<15f} {:<1d}'.format(ar, ba, ec,en,bar,i+1))
#    if prop[i].area > 50 and prop[i].area < 1000:
    if ec >= 0.998:
        lf = i+1
        print('cumplio')
fibra = li==lf           
     
#plt.figure()
#plt.imshow(ima-im_p,cmap='gray_r')
#plt.show()  
plt.figure()
plt.imshow(thin(fibra),cmap='gray_r')
plt.title('thin')
plt.show()  
plt.figure()
plt.imshow(skeletonize(fibra),cmap='gray_r')
plt.title('skel')
plt.show()  

#plt.figure()
#plt.imshow(li,cmap='gray_r')
#plt.show()  
#%%
from skimage.filters import sato
ff = 1 #361, 427
ni = '{:04d}'.format(ff)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\ID_0_C1S0004\ID_0_C1S000400'+ni+'.tif')
bina = np.mean(ima-im_p) - 2.2 * np.std(ima-im_p) #63
#bina = -75
im = ima-im_p
t1 = time()
sat = sato(im, mode='constant')
t2 = time()
print('sato',t2-t1)
t1 = time()
sat2 = sato(im[950:,500:700], mode='constant')
t2 = time()
print('sato chico',t2-t1)

plt.figure()
plt.imshow(im[:,:],cmap='gray_r')
plt.show()  
plt.figure()
plt.imshow(sat,cmap='gray_r')
plt.title('Sato')
plt.show()  
plt.figure()
plt.imshow(sat2,cmap='gray_r')
plt.title('Sato')
plt.show()  

#1024x1024: 1.5 seg aprox
#324x400: 0.17 seg aprox
#74x200: 0.012 seg aprox
#%%
#posibles numeros a ver: convex_area, eccentricity, euler_number, filled_area
#Pareciera que el numero de euler ( Euler characteristic of region. Computed as number of objects (= 1)
#    subtracted by number of holes (8-connectivity) ) es siempre 0 o 1,
# y la excentricidad mayor a 0.999 
#no se que tan general sea esto, en esta fibra la excentricidad pareciera distinguir la fibra del ruido
#pero tal vez es porque es muy recta, y el número de euler no es suficiente
# (cada tanto pasa que el ruido tiene un número de euler 1 o 0 )

#la bbox_area/area pareciera funcionar también, podría ser nuevamente porque la fibra es muy recta
#%%

