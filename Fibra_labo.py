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
from PIL import Image
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
plt.imshow(im_p,cmap='gray_r')
plt.show()
#%%
imags = []
t1 = time()
for i in range(1,100):
    ni = '{:04d}'.format(i)
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\ID_0_C1S0004\ID_0_C1S000400'+ni+'.tif')
    imags.append(ima)
t2 = time()
print(t2-t1)
#%%
ff = 2
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
fib = []
for i in range(15):
    bina = np.mean(imags[i]-im_p) - 63
    fibrass,bbs = spl.encuentra_fibra([imags[i]-im_p],binariza=bina,connec=0)
    try:
        fibra,bb = fibrass[0], bbs[0]
        fib.append(fibra)    
    except: 
        print('no encontró',i)
#%%
for i in range(5):
    plt.figure()
    plt.imshow(fib[i])
    plt.show()

#%%
ff = 2
#ni = '{:04d}'.format(ff)
#ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\ID_0_C1S0004\ID_0_C1S000400'+ni+'.tif')
bina = np.mean(imags[ff]-im_p) - 63
#bina = -69
#print(bina, np.mean(imags[ff]-im_p), np.std(imags[ff]-im_p))
im = (imags[ff]-im_p)<bina
     
fibra = remove_small_objects(im, connectivity=1)
li = label(fibra)
prop = regionprops(li)
for i in range(np.max(li)):
    if prop[i].area > 50 and prop[i].area > 1000:
        lf = i
fibra = li==lf           
     
plt.figure()
plt.imshow(im,cmap='gray_r')
plt.show()  
plt.figure()
plt.imshow(fibra,cmap='gray_r')
plt.show()  




