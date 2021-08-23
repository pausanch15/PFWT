#Acá levanto los archivos que generé en estadis_limpio_p.py y hago cosas.
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep, splprep
import Func_Genera_Fibras_2 as gf
import Func_Splines as spl
from time import time
from scipy import interpolate
import pickle
from scipy import optimize
import scipy
import statsmodels.api as sm
plt.ion()

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

#Las que difieren mas de 5 píxeles son estas
# r = [446, 535, 616, 640, 1006, 1220, 1324, 1361, 1586, 1788, 1881, 1937, 2045, 2089, 2163, 2421, 2507, 2510, 2553, 2604, 2662, 2755, 287, 2811, 2831]
# 
# ff = 446
# 
# t_spl = np.linspace(0, 1, 10000)
# xf, yf = splev(t_spl, s[ff+500])
# yo, xo = splev(t_spl, so[ff+500])
# xf,yf,z = uQuery([xf,yf],u,steps).T
# xo,yo,z = uQuery([xo,yo],u,steps).T
# curvo = gf.curva(xo,yo)
# curvf = gf.curva(xf,yf)
# print(curvo,curvf)
# 
# plt.figure()
# plt.set_cmap('gray')
# plt.imshow(imagenes[ff][0])
# plt.plot(xf, yf, 'r-')
# plt.plot(xo[::1], yo[::1], 'g-')
#plt.plot(xf-xo[::1], 'r-')
#plt.plot(yf-yo[::1], 'g-')

#Hacemos los histogramas y dibujamos normales sobre las diferencias en x y en y.
print(f'Para la diferencia en x, la media es de {np.mean(dx)} y el desvío {np.std(dx)}.')
print(f'Para la diferencia en y, la media es de {np.mean(dy)} y el desvío {np.std(dy)}.')
print(f'Para la diferencia en la curvatura, la media es de {np.mean(curv)} y el desvío {np.std(curv)}.')

z_x = (dx - np.mean(dx)) / (np.std(dx))
z_y = (dy - np.mean(dy)) / (np.std(dy))

plt.figure()
plt.grid()
plt.hist(z_x, bins=150, color='#eb5600', label='x', alpha=1, density=True, stacked=True, edgecolor='black', linewidth=0.5)
plt.legend()
plt.title('Diferencias en x')

plt.figure()
plt.grid()
plt.hist(z_y, bins=150, color='#1a9988', label='y', alpha=1, density=True, stacked=True, edgecolor='black', linewidth=0.5)
plt.legend()
plt.title('Diferencias en y')


# plt.figure()
# plt.hist(dx, bins='auto', color='black', label='x', alpha=1, density=True, stacked=True)
# pdf = normal_dist(dx, np.mean(dx), np.std(dx))
# plt.plot(dx, pdf, 'r.', label='Distribución Normal')
# plt.legend()
# plt.title('Diferencias en x')
# 
# plt.figure()
# plt.hist(dy, bins='auto', color='black', label='y', alpha=1, density=True, stacked=True)
# pdf = normal_dist(dy, np.mean(dy), np.std(dy))
# plt.plot(dy, pdf, 'r.', label='Distribución Normal')
# plt.legend()
# plt.title('Diferencias en y')
# plt.show()
# 
# plt.figure()
# plt.hist(curv, bins='auto', density=True, stacked=True, color='black', alpha=1)
# plt.title('Curvatura')
# plt.show()

#Hacemos el kstest para ver si nuestras distribuciones para las diferencias en x y en y se parecen a una distribución gaussiana
#Para x
# est_x, pv_x = scipy.stats.kstest(dx, 'norm')
# print(f'El estadístico obtenido para x es {est_x} con un p-valor de {pv_x}.')
# 
# #Para x
# est_y, pv_y = scipy.stats.kstest(dy, 'norm')
# print(f'El estadístico obtenido para y es {est_y} con un p-valor de {pv_y}.')

#Hacemos el qq plot
#https://www.statsmodels.org/stable/generated/statsmodels.graphics.gofplots.qqplot.html
# sm.qqplot(np.array(dx), loc=np.mean(dx), scale=np.std(dx), line='45')
# sm.qqplot(np.array(dy), loc=np.mean(dy), scale=np.std(dy), line='45')
