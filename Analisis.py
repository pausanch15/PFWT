import numpy as np
import matplotlib.pyplot as plt
import Func_Splines as spl
import Continuacion_Fourier as cf
from skimage.io import imread
from scipy.interpolate import splev
#%%
gris = np.zeros((1024,1024))
for i in range(1,301):
    ni = '{:04d}'.format(i)
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\Grises\ID_0_C1S000600'+ni+'.tif')
    ima = np.array(ima,dtype='float')
    gris += ima
gris = gris/300

ref = np.zeros((1024,1024))
for i in range(1,301):
    ni = '{:04d}'.format(i)
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\Referencia\ID_0_C1S000300'+ni+'.tif')
    ima = np.array(ima,dtype='float')
    ref += ima
ref = ref/300
#%%
plt.figure()
plt.imshow(ref-gris)
plt.show()


#%%
from time import time
t1 = time()
num = 1 #568, 165
ni = '{:04d}'.format(num)
im = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\FTP_fibra\ID_0_C1S000500'+ni+'.tif')
im = np.array(im,dtype='float')

mul = 2.46
fibra, bb, fib_an = spl.recupera_fibra(im,gris,std_mul=mul)
tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=True)
tramos = spl.ordenar_fibra(tramos)
try:
    curv,spline = spl.pegar_fibra(tramos,bordes,window=15,s=16)
except: print('Jajaja')
tm = time()
#t_spl = np.linspace(0, 1, 10000)
#xf, yf = splev(t_spl, spline)

imfc, ffc = cf.hacer_fou_cont(im,fibra,bb,ventana=150,mmx=2,mmy=2)
tm2 = time()
thx,thy, ns = 0.5, 80, 0.5
dph, ft, gf = cf.dphase_2d(imfc-gris,im,thx,thy,ns,inde=9)
t2 = time()
tm2 - tm, t2-tm2, tm-t1
#%%
#imfc, ffc = cf.hacer_fou_cont(im,fibra,bb,ventana=150,mmx=2,mmy=2)
thx,thy, ns = 0.5, 80, 0.5
dph, ft, gf = cf.dphase_2d(imfc-gris,ref-gris,thx,thy,ns,inde=9)
lim = 10
plt.figure()
plt.imshow(dph[lim:-lim,lim:-lim])
plt.colorbar()
plt.show()
#plt.figure()
#plt.imshow(imfc-gris,cmap='gray')
#plt.show()
plt.figure()
plt.imshow(im-gris,cmap='gray')
plt.plot(xf, yf, 'r-')
plt.colorbar()
plt.show()
#%%
