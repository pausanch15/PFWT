#Ejemplo de como funciona el codigo en general
import Func_Splines as spl
import Func_Genera_Fibras as gf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import thin, skeletonize, remove_small_objects, dilation
from scipy.interpolate import CubicSpline, splev, splrep, splprep
from scipy.signal import convolve2d, savgol_filter
from itertools import permutations
import imageio

#Generamos n fibras
# Buenas: 12, 15
np.random.seed(15)
n = 5
imagenes, fib = gf.crear_im_fibra(n+1, ruido=0.0015, fondo=0.05, salto=15, drift=[0,0])

#Probamos con la primera de las fibras generadas
fibras = spl.encuentra_fibra(imagenes, binariza=113)
ff = 0
fibra = fibras[ff]
tramos,bordes = spl.cortar_fibra(fibra)
tramos = spl.ordenar_fibra(tramos)
curv,spline = spl.pegar_fibra(tramos, bordes, window=17, s=8)

t_spl = np.linspace(0,1,10000)
xf,yf = splev(t_spl,spline)

#Vemos la imagen de la fibra
plt.figure()
plt.set_cmap('gray')
plt.imshow(imagenes[ff])
plt.show(block=True)

#Y la fibra que logramos obtener de la imagen
plt.figure()
plt.plot(xf,yf,'r-')
plt.show()

yo,xo = splev(t_spl,fib[ff])

#Vemos la diferencia entre la fibra generada y la fibra obtenida a partir de la imagen
#Primero x e y por separado
plt.figure()
plt.hist(xf-xo,bins=100,color='blue',label='x')
plt.hist(yf-yo,bins=100,color='red',label='y')
plt.legend()
plt.show()

#Ahora la suma
plt.figure()
plt.hist((xf-xo)+(yf-yo),bins=50)
plt.title('(xf-xo)+(yf-yo)')
plt.show()

#Repetimos para las n imagenes
fibras = spl.encuentra_fibra(imagenes, binariza=113)
splines = []
for ff in range(len(imagenes)):
    print(ff,end=' ')
    fibra = fibras[ff]
    tramos,bordes = spl.cortar_fibra(fibra)
    tramos = spl.ordenar_fibra(tramos)
    curv,spline = spl.pegar_fibra(tramos,bordes,window=17,s=8)
    splines.append(spline)

dx, dy, dxdy = [], [], []
t_spl = np.linspace(0,1,10000)
for i in range(len(fib)):
    xf,yf = splev(t_spl,splines[i])
    yo,xo = splev(t_spl,fib[i])
    if np.max(np.abs(xf-xo)) > 50:
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

#No me queda claro que es esto, pero lo dejo.
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

