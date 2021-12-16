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
for num in range(1,20):
    if num%10 == 0: print(num, end=' ')
    ni = '{:04d}'.format(num)
    ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0001/ID_0_C1S000100'+ni+'.tif')
    # ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
    im = np.array(ima,dtype='float')
    dph, ft, gf = cf.dphase_2d(im,ref-gris,thx,thy,ns,inde=9)
    dphs[num-1] = dph

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
#Hago la animación
from mpl_toolkits.mplot3d import Axes3D
x = np.arange(0,1024,1)
xx, yy = np.meshgrid(x,x)

from matplotlib import animation
i,li = 3, 10
at, xt, yt = alturas[:,li:-li:i,li:-li:i], xx[li:-li:i,li:-li:i], yy[li:-li:i,li:-li:i] 

def update_plot(frame_number, at, plot):
    plot[0].remove()
    plot[0] = ax.plot_surface(xt, yt, (at[frame_number]-np.mean(at[frame_number])), cmap="magma")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

fps = 20 # frame per sec
frn = 20 # frame number of the animation

plot = [ax.plot_surface(xt, yt, at[0]-np.mean(at[0]), color='0.75', rstride=10, cstride=10)]
ax.set_zlim(2.5,-2.5)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(at, plot), interval=1000/fps)
