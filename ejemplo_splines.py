#Ejemplo para la presentacion de como pasamos de la imagen original de la fibra a la fibra interpolada.
import Func_Splines as spl
import Func_Genera_Fibras as gf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import thin, skeletonize, remove_small_objects
from scipy.interpolate import CubicSpline, splev, splrep, splprep
from scipy.signal import convolve2d, savgol_filter
from itertools import permutations
import imageio
#%%
#Veo con una fibra generada
# Buenas: 12, 15
np.random.seed(12)
n = 40
imagenes, fib = gf.crear_im_fibra(n+1,ruido=0.0015,fondo=0.05,salto=5,drift=20,largo_fibra=300)

imageio.mimsave('C:\\Users\\tomfe\\Documents\\TOMAS\\Facultad\\Laboratorio 6\\Github\\probando_12.gif',imagenes,fps=12)
#%%
fibras = spl.encuentra_fibra(imagenes,binariza=105)
#%%
ff = 36
fibra = fibras[ff]
tramos,bordes = spl.cortar_fibra(fibra,cortar_ruido=False)
tramos = spl.ordenar_fibra(tramos)
curv,spline = spl.pegar_fibra(tramos,bordes,window=21,s=10)

t_spl = np.linspace(0,1,10000)
xf,yf = splev(t_spl,spline)

#plt.figure()
#plt.set_cmap('gray')
#plt.imshow(imagenes[ff])
#plt.show()
#plt.figure()
#plt.imshow(fibra)    
#plt.plot(xf,yf,'r-')
#plt.show()

#plt.gca().invert_yaxis()
#for tr in range(len(tramos)):
#    x,y = tramos[tr][:,0],tramos[tr][:,1]
#    plt.plot(x,y,'o')#,color='grey')

yo,xo = splev(t_spl,fib[ff])

#plt.figure()
#plt.plot(xo,yo,'g-') 
#plt.plot(xf,yf,'r-')
#plt.show()
plt.figure()
plt.plot(t_spl,xo,'r--')
plt.plot(t_spl,xf,'r-')
#plt.plot(t_spl,yf,'g-')
#plt.plot(t_spl,yo,'g--')
plt.grid(True)
plt.show()

np.max(np.abs(xf-xo))
#plt.figure()
#plt.hist(xf-xo,bins=100,color='blue',label='x')
#plt.hist(yf-yo,bins=100,color='red',label='y')
#plt.legend()
#plt.show()
#plt.figure()
#plt.hist((xf-xo)+(yf-yo),bins=50)
#plt.title('(xf-xo)+(yf-yo)')
#plt.show()
#%%
fibras = spl.encuentra_fibra(imagenes,binariza=105)
splines = []
for ff in range(len(imagenes)):
    print(ff,end=' ')
    fibra = fibras[ff]
    tramos,bordes = spl.cortar_fibra(fibra,cortar_ruido=False)
    tramos = spl.ordenar_fibra(tramos)
    curv,spline = spl.pegar_fibra(tramos,bordes,window=21,s=10)
    splines.append(spline) 
#%%
dx, dy, dxdy = [], [], []
t_spl = np.linspace(0,1,10000)
for i in range(len(fib)):
    xf,yf = splev(t_spl,splines[i])
    yo,xo = splev(t_spl,fib[i])
    if np.max(np.abs(xf-xo)) > 9:
        xo = xo[::-1]
        yo = yo[::-1]
    dx = dx + list(xf-xo)
    dy = dy + list(yf-yo)
    dxdy = dxdy + list( (xf-xo)+(yf-yo) )
    
plt.figure()
plt.hist(dx,bins=100,color='blue',label='x',alpha=0.5)
plt.hist(dy,bins=100,color='red',label='y',alpha=0.5)
plt.legend()
plt.show()
plt.figure()
plt.hist(dxdy,bins=100)
plt.title('(xf-xo)+(yf-yo)')
plt.show()

np.mean(dxdy), np.std(dxdy)
#%%

##%%
#lista_im = []
#for frames in range(len(imagenes)):
#    print(frames)
#    imt = imagenes[frames] < 113
#    iml = label(imt)
#    prop_reg = regionprops(iml)
#    for i in range(len(prop_reg)):
#        if prop_reg[i].area > 100:
#            fibra = iml==i+1
#    fibra = skeletonize(fibra)
#
#    tramos,bordes = spl.cortar_fibra2(fibra)
#    
##    for tr in range(len(tramos)):
##        x,y = tramos[tr][:,0],tramos[tr][:,1]
##        plt.plot(x,y,'o',color='grey')
#    
#    tramos = spl.ordenar_fibra(tramos)
#    t,curv,xf,yf = spl.pegar_fibra(tramos,bordes)
#    plt.ioff()
#    plt.figure()
#    plt.imshow(imagenes[frames])
#    plt.plot(xf,yf,'r-')
#    plt.set_cmap('gray')
##    plt.xlim(100,300)
##    plt.ylim(650,825)
##    plt.gca().invert_yaxis()
##    plt.grid(True)
##    plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
#    plt.savefig('imageni.png',bbox_inches='tight')
#    lista_im.append(imageio.imread('imageni.png'))
##%%
#imageio.mimsave('C:\\Users\\tomfe\\Documents\\TOMAS\\Facultad\\Laboratorio 6\\fibrayspline2.gif',lista_im,fps=12)

