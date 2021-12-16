#Librerías
import numpy as np
import matplotlib.pyplot as plt
import Func_Splines as spl
import Continuacion_Fourier as cf
from skimage.io import imread
from scipy.interpolate import splev
from time import time
import h5py 
plt.ion()

#%%
#Traigo imagenes
gris = np.zeros((1024, 1024))
for i in range(1, 16):
    ni = '{:04d}'.format(i)
    ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
    # ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0003\ID_0_C1S000300'+ni+'.tif')
    ima = np.array(ima, dtype='float')
    gris += ima
gris = gris/15

ref = np.zeros((1024, 1024))
for i in range(1, 14):
    ni = '{:04d}'.format(i)
    ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0002/ID_0_C1S000300'+ni+'.tif')
    # ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0002\ID_0_C1S000300'+ni+'.tif')
    ima = np.array(ima, dtype='float')
    ref += ima
ref = ref/13

#%%
#Para pasar de diferencia de fase a altura, necesitamos la longitud de onda del patrón de rayas
from scipy.signal import find_peaks
asd = ref - gris
xp = np.arange(0,1024,1)
x = 0.02086 * xp
n = 200
lin = asd[n,:] - np.mean(asd[n,:])

a, _ = find_peaks(lin,distance=10) #hay 82 picos, o sea, 81 longitudes de onda
d = (1014-9) / 81 * 0.02086
w = 2*np.pi/d
rec = np.cos(w*x + 1.5) * 75

#%%
#Saco altura para las imagenes que tienen la fibra
#Hago ftp
thx,thy, ns = 0.25, 45, 0.75 #0.5, 80, 0.5
t1 = time()
dphs = np.zeros((100,1024,1024))
for num in range(1, 20):
    if num%10 == 0: print(num, end=' ')
    ni = '{:04d}'.format(num)
    ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0001/ID_0_C1S000100'+ni+'.tif')
    # ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
    im = np.array(ima,dtype='float')
    dph, ft, gf = cf.dphase_2d(im,ref-gris,thx,thy,ns,inde=9)
    dphs[num-1] = dph
    del(ni, ima, im, dph)

#Defino los valores de L y D (los medimos en el labo)
w = 2*np.pi/d
L, D = 79.6, 20.3

#El array con las alturas
alt = (L*dphs) / (dphs - w*D) 

#Y le resto la altura media a cada frame
alturas = []
for frame in alt:
    alturas.append(frame - np.mean(frame))

alturas = np.array(alturas)
del(alt)

#%%
#Ahora los hsitogramas
#Primero de las velocidades
v_un_frame = np.shape((alturas[1] - alturas[0]).flatten())[0] #Cantidad de velocidades por frame
n_frames = len(alturas)
v = np.zeros(v_un_frame*n_frames) #array que va a tener la velocidad de cada punto de todos los frames

for i, im in enumerate(alturas):
    i = i + 1
    if i == len(alturas):
        break
    velocidad = (250*(alturas[i]-im)).flatten()
    inic =  (i-1)*v_un_frame
    fin = i*v_un_frame
    v[inic:fin] = velocidad
    del(velocidad, inic, fin)

plt.hist(v, bins=100, density=True, edgecolor="black")
plt.grid()
plt.title('Histograma de Velocidades')
plt.show()

#Ahora con las aceleraciones
a = np.zeros(v_un_frame*n_frames)

for i in range(1, len(alturas)-1):
    aceleracion = (250**2)*(alturas[i+1]-2*alturas[i]+alturas[i-1])
    aceleracion = aceleracion.flatten()
    inic =  (i-1)*v_un_frame
    fin = i*v_un_frame
    a[inic:fin] = aceleracion
    del(aceleracion, inic, fin)

#Hago el histograma de aceleración
plt.hist(v, bins=100, density=True, edgecolor="black")
plt.grid()
plt.title('Histograma de Aceleración')
plt.show()
