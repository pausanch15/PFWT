#Trato de hacer el gráfico de cuánto tarda el algoritmo en analizar alguna cantidad de imágenes.
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep, splprep
import Func_Genera_Fibras_2 as gf
import Func_Splines as spl
from time import time
from scipy import interpolate
import pickle
from scipy.stats import norm
from scipy import optimize
from scipy.optimize import curve_fit
plt.ion()

#%%
#La función que ordena los splines
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

#La función que calcula la curvatura
def curvatura_pap(x,y):
    splin, u = splprep([x,y],s=0)
    spline_1d = splev(u,splin,der=1)
    spline_2d = splev(u,splin,der=2)
    curv = np.abs(spline_2d[0] * spline_1d[1] - spline_1d[0] * spline_2d[1]) / \
                 ((spline_1d[0])**2 + (spline_1d[1])**2)**(3/2)
    return curv 

#Empezamos a crear las imágenes y analizarlas     
n_int = 1
n_fib = 3000
curvs, fibras, tttr, brrr = [],[],[],[]
tttr, brrr = [],[]
np.random.seed(12)

t = [0.0]
n_im = [0]
ti = time()

for i in range(n_fib):
    if i%100 == 0:
        print(f'\nVa por la imagen {i}.')

    im, sss = gf.genera_im_dinamica(frames=n_int, n_fibras=2, alpha=0.1,N=1000,Nt=8,curvatura=100)
    tint = time()
    fibrass,bbs = spl.encuentra_fibra(im,binariza=70)
    fibra,bb = fibrass[0], bbs[0]
    fibras.append(fibra)
    tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=False)
    tramos = spl.ordenar_fibra(tramos)
    tttr.append(tramos)
    brrr.append(bordes)

    try:
        curv,spline = spl.pegar_fibra(tramos,bordes,window=17,s=10)
        curvs.append(curv)
        del(curv,spline)
    except UnboundLocalError:
        curvs.append('Nan')
        print(f'\nTiró UnboundLocalError: {i}.')

    tf = time()
    t.append(tf-ti)
    n_im.append(i)
    
    del(im, sss, fibrass, bbs, fibra, bb, tramos, bordes)

#Medio al pedo si ya tenía las listas, pero guardo lo obtenido en csv para el futuro
np.savetxt('tiempos.csv', t, delimiter=',')
np.savetxt('n_im.csv', n_im, delimiter=',')

plt.plot(t, n_im, 'k.')
plt.xlabel('Tiempo (s)')
plt.ylabel('Número de Imágenes')
plt.grid()
plt.show()

#%%
#Levanto los archivos como arrays
t = np.genfromtxt('tiempos.csv', delimiter=',', dtype=None)
n_im = np.genfromtxt('n_im.csv', delimiter=',', dtype=None)

#%%
#Tomo cada tres valores de los array, promedio y saco el desvío
t_m = []
t_std = []
n_im_m = []
n_im_std = []
for i in range(len(t)):
    if t[i*3:(i*3)+3][-1] == t[-1]: break
    t_m.append(np.mean(t[i*3:(i*3)+3]))
    t_std.append(np.std(t[i*3:(i*3)+3], ddof=1))
    n_im_m.append(np.mean(n_im[i*3:(i*3)+3]))
    n_im_std.append(np.std(n_im[i*3:(i*3)+3], ddof=1))

np.array(t_m)
np.array(n_im_m)
np.array(t_std)
np.array(n_im_std)

#%%
#Ploteo
plt.plot(n_im_m, t_m, 'k.')
plt.ylabel('Tiempos Promediados (s)')
plt.xlabel('Número de Imágenes Promediados')
plt.grid()
plt.show()

#Hago el ajuste
def recta(x, m, b):
    return m*x + b

popt, pcov = curve_fit(recta, n_im_prom, t_prom, sigma=t_std)

x = np.linspace(0, np.max(n_im_m), 3000)

plt.plot(n_im_prom, t_prom, 'k.')
plt.plot(x, recta(x, *popt), 'r--')
plt.errorbar(n_im_prom, t_prom, yerr=n_im_std, ecolor='gray', fmt='none')
plt.show()
