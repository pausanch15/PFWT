#A ver si se puede hacer estadística.
#Esta vez no vamos a crear una lista de n imégenes y despues analizar todas, porque esto llena la memoria. En vez, vamos a crear una imagen, analizarla, después crear otra, analizarla, y así sucesivamente.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import splev, splrep, splprep
from skimage.morphology import thin, skeletonize, remove_small_objects, binary_dilation, dilation
from skimage.util import random_noise
from skimage.filters import gaussian
import Func_Genera_Fibras_2 as gf
import Func_Splines as spl
from time import time
from scipy import interpolate
from itertools import permutations
import pickle
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
#%%
#Empezamos a crear las imágenes y analizarlas

# data = [las cosas a guardar]
# with open("archivoaguardar.dat", "wb") as fp:
    # pickle.dump(data, fp)
# with open("archivodondeguarde.dat", "rb") as fp:
    # data = pickle.load(fp)

j = 0

with open("splines.dat", "wb") as sp, open("splineso.dat", "wb") as spo:
    while j < 5:
    
        print(f'Vamos por la tanda {j}.')
        
        n_int = 1
        n_fib = 500
        imagenes = []
        curvs, fibras, tttr, brrr = [],[],[],[]
        tttr, brrr = [],[]
        np.random.seed(12)

        ti = time()
        for i in range(n_fib):
            im, sss = gf.genera_im_dinamica(frames=n_int, n_fibras=2, alpha=0.1,N=1000,Nt=8,curvatura=150)
            imagenes.append(im[0])
            pickle.dump(sss[0], spo)
            fibrass,bbs = spl.encuentra_fibra(im,binariza=50)
            fibra,bb = fibrass[0], bbs[0]
            fibras.append(fibra)
            tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=False)
            tramos = spl.ordenar_fibra(tramos)
            tttr.append(tramos)
            brrr.append(bordes)

            try:
                curv,spline = spl.pegar_fibra(tramos,bordes,window=17,s=10)
                pickle.dump(spline, sp)
                curvs.append(curv)
                del(curv,spline)
            except UnboundLocalError:
                pickle.dump('Nan', sp)
                curvs.append('Nan')
                print('Las imégenes en las que tira UnboundLocalError son:')
                print(i)
            del(im,sss,fibrass,bbs,fibra,bb,tramos,bordes)

        #Guardo las imagenes como arrays
        for i, im in enumerate(imagenes):
            np.savez(f'{j}_im_{i}.npz', im)

        j += 1
        tf = time()
        print(f'\nTarda {tf-ti} segundos en crear y analizar {n_fib} imagenes.')

#%%
#Esta línea sirve para armarnos una lista con las imagenes que guardamos antes. A cada elemento de imagenes se le puede hacer un imshow y se ven las imagenes.
archivos = [f'{j}_im_{i}.npz' for j in range(5) for i in range(500)]
imagenes = [np.load(archivo)["arr_0"] for archivo in archivos]

#%%
#Ahora evaluamos los splines originales y los obtenidos en un mismo array de puntos, para poder comparar las distancias entre los x y lo y y las curvaturas. Siempre comparamos original vs recuperado.
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

ti = time()
u = np.linspace(0,1,1000)
steps = 50000
dx, dy, curv = [], [], []
t_spl = np.linspace(0,1,10000)
for i in range(len(fibras)):
    if i in [112,479,541,624,673,782]: continue
    if i in [5,14,15,111,214,248,285,313,383,404,500,521,571,703,733,755,960]: continue #estan bien, los saco para ver
#    print(i)
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
    if np.max(np.abs(yf-yo)) > minn or np.max(np.abs(xf-xo)) > minn: print(i,end=' ')
    dx = dx + list(xf-xo)
    dy = dy + list(yf-yo)
    curv = curv+ list(curvf-curvo)
    del(xf,xo,yo,yf,z)
tf = time()
print(f'Tarda {tf-ti} segundos en evaluar los splines originales y recuperados de {n_fib} imagenes.')
#%%
# # Hacemos los histogramas
# ti = time()
# histx, limx = np.histogram(dx, bins='auto')
# histy, limy = np.histogram(dy, bins='auto')
# tf = time()
# print(f'Tarda {tf-ti} segundos en generar el histograma para {len(imagenes)} imágenes.')

# ti = time()
# plt.figure()
# plt.plot(histx, color='blue', label='x', alpha=0.5)
# plt.plot(histy, color='red', label='y', alpha=0.5)
# plt.legend()
# plt.show()
# tf = time()
# print(f'Tarda {tf-ti} segundos hacer el gráfico de los histogramas')

print(f'Para la diferencia en x, la media es de {np.mean(dx)} y el desvío {np.std(dx)}.')
print(f'Para la diferencia en y, la media es de {np.mean(dy)} y el desvío {np.std(dy)}.')
print(f'Para la diferencia en la curvatura, la media es de {np.mean(curv)} y el desvío {np.std(curv)}.')

# plt.figure()
# plt.hist(dx, bins='auto', color='blue', label='x', alpha=0.5, density=True, stacked=True)
# plt.hist(dy, bins='auto', color='red', label='y', alpha=0.5, density=True, stacked=True)
# plt.legend()
# plt.title('Diferencias en x y en y')
# plt.show()
# 
# plt.figure()
# plt.hist(curv, bins='auto', density=True, stacked=True, color='blue', alpha=0.5)
# plt.title('Curvatura')
# plt.show()
# 
# del(dx, dy, curv)
