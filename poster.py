#Armo las figuras para los posters
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import thin
from scipy.interpolate import splev, splrep, splprep
import Func_Genera_Fibras_2 as gf
import Func_Splines as spl
from time import time
from scipy import interpolate
import pickle
from scipy import optimize
import scipy
import statsmodels.api as sm
import matplotlib.ticker as ticker
from skimage.feature import corner_harris
from matplotlib.colors import ListedColormap
import sys
from skimage.morphology import dilation
plt.ion()

#%%
#Fibra con el thin y binarizada
image = Image.open('imagenprueba.png') #cargo la imagen
ima = np.asarray(image) #la convierto en numpy array
im = ima[:,:,0] #eligo quedarme con el color rojo
imc = im[:, 100:800] #me quedo entre las lineas 100 y 800 (quito las turbinas)
imt = imc<200 #pongo el threshold en 200 para binarizar la imagen
li = label(imt) # etiqueto cada región contigua de pixeles llenos 
fibra = li==6  #me quedo con la fibra (vimos graficamente que la fibra tenia la label 6)
fibra = thin(fibra)
plt.imshow(fibra)
plt.colorbar() #para que muestre el rango en el que van los colores
plt.set_cmap('gray') #para que la imagen sea en blanco y negro
plt.show()

#%%
#Figura que marca los bordes y los nudos
plt.rc("text", usetex=True)
plt.rc('font', size=19)
plt.figure()
# plt.plot([324, 297], [427, 566], '.', color='#ffc000', markersize=4, label='Bordes')
# plt.plot([462, 474], [496, 491], '.', color='#70ad47', markersize=4, label='Nudos')
plt.plot([36, 10], [17, 155], '.', color='#ffc000', markersize=4, label='Bordes')
plt.plot([175, 187], [86, 821], '.', color='#70ad47', markersize=4, label='Nudos')
plt.xticks([])
plt.yticks([])
plt.imshow(fibra[410:577, 287:600])
plt.set_cmap('gray_r') #para que la imagen sea en blanco y negro
plt.legend()
plt.show()
# plt.savefig('bordes_nudos.png', dpi=300)

#%%
#Figura de la fibra cortada con cada sección de un color
corners = corner_harris(fibra, k=0.0005)
corners = corners>3.8
fibra_cortada = dilation(fibra) + ((-1)*corners)
fibra_cortada = fibra_cortada > 0.1
secciones, cantidad_secciones = label(fibra_cortada, return_num=True)

my_cmap = ListedColormap(['white', '#ffc000', '#70ad47', '#ed7d31', 'black'])

fig, ax = plt.subplots()
cbr = ax.imshow(secciones[400:600, 290:600], cmap=my_cmap, vmin=0, vmax=4)
# fig.colorbar(cbr)
ax.xaxis.set_major_locator(ticker.NullLocator())
ax.yaxis.set_major_locator(ticker.NullLocator())

plt.savefig('tramos.png', dpi=300)

#%%
#Figura que tiene todo lo de arriba junto
fig, axs = plt.subplots(1, 3, figsize=(20, 10))

axs[0].imshow(ima)
axs[1].imshow(fibra)
axs[1].plot([324, 297], [427, 566], '.', color='#ffc000', markersize=4, label='Bordes')
axs[1].plot([462, 474], [496, 491], '.', color='#70ad47', markersize=4, label='Nudos')
axs[1].legend()
axs[2].imshow(secciones[400:600, 290:600], cmap=my_cmap, vmin=0, vmax=4)

for ax in axs:
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())

plt.savefig('algo.png', dpi=300)

#%%
#Histogramas de diefrencias en x e y
#Levanta imágenes ya obtenidas
#Funciones
def uQuery(pts,u,steps=100,projection=True): 
    u = np.clip(u,0,1) 
    x,y = pts[0],pts[1]
    z = np.zeros_like(x)
    cv = np.vstack((x,y,z)).T
    
    samples = np.linspace(0,1,steps)
    tck,u_=interpolate.splprep(cv.T,s=0.0)
    p = np.array(interpolate.splev(samples,tck)).T     

    p_= np.diff(p,axis=0) 
    m = np.sqrt((p_*p_).sum(axis=1)) 
    s = np.cumsum(m)
    s/=s[-1]

    s = np.insert(s,0,0) 
    i0 = (s.searchsorted(u,side='left')-1).clip(min=0) 
    i1 = i0+1 

    if projection:
        return ((p[i1]-p[i0])*((u-s[i0])/(s[i1]-s[i0]))[:,None])+p[i0]

    mod = (((u-s[i0])/(s[i1]-s[i0]))/steps)+samples[i0]
    return np.array(interpolate.splev(mod,tck)).T 

def curvatura_pap(x,y):
    splin, u = splprep([x,y],s=0)
    spline_1d = splev(u,splin,der=1)
    spline_2d = splev(u,splin,der=2)
    curv = np.abs(spline_2d[0] * spline_1d[1] - spline_1d[0] * spline_2d[1]) / \
                 ((spline_1d[0])**2 + (spline_1d[1])**2)**(3/2)
    return curv 

def normal_dist(x , mean , sd):
    # prob_density = (np.pi*sd) * np.exp(-0.5*((x-mean)/sd)**2)
    prob_density = (1/(2*np.pi*(sd**2))**(0.5)) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

#Levantamos los archivos con los splines
s = []
so = []
with open("splines.dat", "rb") as sp, open("splineso.dat", "rb") as spo:
    try:
        while True:
            spli = pickle.load(sp)
            splio = pickle.load(spo)
            s.append(spli)
            so.append(splio)
    except EOFError:
        pass

#Y las imágenes
# archivos = [f'im_{i}.npz' for i in range(len(s))]
# imagenes = [np.load(archivo)["arr_0"] for archivo in archivos]

#Calculamos dx, dy y la curvatura
#Si queremos sacar las imágenes más ruidosas, usamos esto.
r = [446, 535, 616, 640, 1006, 1220, 1324, 1361, 1586, 1788, 1881, 1937, 2045, 2089, 2163, 2421, 2507, 2510, 2553, 2604, 2662, 2755, 2807, 2811, 2831]

ti = time()
u = np.linspace(0,1,1000)
steps = 50000
dx, dy, curv = [], [], []
t_spl = np.linspace(0,1,10000)
for i in range(len(s)):
    if i%100 == 0:
        print(f'\nVa por la imagen {i}.')
    if i in r: continue
    xf,yf = splev(t_spl,s[i])
    yo,xo = splev(t_spl,so[i])
    xf,yf,z = uQuery([xf,yf],u,steps).T
    xo,yo,z = uQuery([xo,yo],u,steps).T
    if np.max(np.abs(xf-xo)) > 20 or np.max(np.abs(yf-yo)) > 20:
        xo = xo[::-1]
        yo = yo[::-1]
    curvf = curvatura_pap(xf,yf)
    curvo = curvatura_pap(xo,yo)
    minn = 5
    dx = dx + list(xf-xo)
    dy = dy + list(yf-yo)
    curv = curv+ list(curvf-curvo)
    del(xf,xo,yo,yf,z)
tf = time()

print(f'Tarda {tf-ti} segundos en evaluar los splines originales y recuperados de {len(s)} imagenes.')

z_x = (dx - np.mean(dx)) / (np.std(dx))
z_y = (dy - np.mean(dy)) / (np.std(dy))

z = [z_x, z_y]
col = ['#ffc000', '#70ad47']
lab = ['x', 'y']
mus = [np.mean(dx), np.mean(dy)]
sigmas = [np.std(dx), np.std(dy)]

plt.rc("text", usetex=True)
plt.rc('font', size=19)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for i, ax in enumerate(axs.flatten()):
    ax.hist(z[i], bins=150, color=col[i], label=lab[i], alpha=1, density=True, stacked=True, edgecolor='black', linewidth=0.5, zorder=3)
    ax.legend()
    ax.grid()
    ax.annotate(f"$\mu =${mus[i]:.2}\n$\sigma =${sigmas[i]:.2}", (7, 0.1))
    
plt.savefig('hist_xy.png', dpi=300)

#%%
plt.rc("text", usetex=True)
plt.rc('font', size=19)
plt.hist(curv, bins=700, color='#ed7d31', label='Curvatura', alpha=1, density=True, stacked=True, edgecolor='black', linewidth=0.5, zorder=3)
plt.xlim((-0.05, 0.05))
plt.annotate(f"$\mu =${np.mean(curv):.2}\n$\sigma =${np.std(curv):.2}", (0.02, 100))
plt.grid()
plt.legend()

plt.savefig('hist_curv.png', dpi=300)

