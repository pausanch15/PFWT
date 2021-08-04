#A ver si se puede hacer estadística.
#Esta vez no vamos a crear una lista de n imégenes y despues analizar todas, porque esto llena la memoria. En vez, vamos a crear una imagen, analizarla, después crear otra, analizarla, y así sucesivamente.
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.animation as animation
from scipy.interpolate import splev, splrep, splprep
#from skimage.morphology import thin, skeletonize, remove_small_objects, binary_dilation, dilation
#from skimage.util import random_noise
#from skimage.filters import gaussian
import Func_Genera_Fibras_2 as gf
import Func_Splines as spl
from time import time
from scipy import interpolate
#from itertools import permutations
plt.ion()
#%%
#La función que ordena los splines
def uQuery(pts,u,steps=100,projection=True): 
#https://stackoverflow.com/questions/34941799/querying-points-on-a-3d-spline-at-specific-parametric-values-in-python
    ''' Brute force point query on spline
        pts = [x,y]
        u      = list of queries (0-1)
        steps  = number of curve subdivisions (higher value = more precise result)
        projection = method by wich we get the final result
                     - True : project a query onto closest spline segments.
                              this gives good results but requires a high step count
                     - False: modulates the parametric samples and recomputes new curve with splev.
                              this can give better results with fewer samples.
                              definitely works better (and cheaper) when dealing with b-splines (not in this examples)
    '''
    u = np.clip(u,0,1) # Clip u queries between 0 and 1
    x,y = pts[0],pts[1]
    z = np.zeros_like(x)
    cv = np.vstack((x,y,z)).T
    # Create spline points
    samples = np.linspace(0,1,steps)
    tck,u_=interpolate.splprep(cv.T,s=0.0)
    p = np.array(interpolate.splev(samples,tck)).T  
    # at first i thought that passing my query list to splev instead
    # of np.linspace would do the trick, but apparently not.    

    # Approximate spline length by adding all the segments
    p_= np.diff(p,axis=0) # get distances between segments
    m = np.sqrt((p_*p_).sum(axis=1)) # segment magnitudes
    s = np.cumsum(m) # cumulative summation of magnitudes
    s/=s[-1] # normalize distances using its total length

    # Find closest index boundaries
    s = np.insert(s,0,0) # prepend with 0 for proper index matching
    i0 = (s.searchsorted(u,side='left')-1).clip(min=0) # Find closest lowest boundary position
    i1 = i0+1 # upper boundary will be the next up

    # Return projection on segments for each query
    if projection:
        return ((p[i1]-p[i0])*((u-s[i0])/(s[i1]-s[i0]))[:,None])+p[i0]

    # Else, modulate parametric samples and and pass back to splev
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

n_int = 1
n_fib = 500
imagenes, fibras = [], []
splines, splineso = [], []
curvs, fibras, tttr, brrr = [],[],[],[]
np.random.seed(12)
t1 = time()
for i in range(n_fib):
    im, sss = gf.genera_im_dinamica(frames=n_int, n_fibras=2, alpha=0.1,N=1000,Nt=8,curvatura=100)
    imagenes.append(im[0])
    splineso.append(sss[0])    
#    if i ==26: continue
    if i%10 == 0: print(i,end=' ')
    fibrass,bbs = spl.encuentra_fibra(im,binariza=70)
    fibra,bb = fibrass[0], bbs[0]
    fibras.append(fibra)
    tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=False)
    tramos = spl.ordenar_fibra(tramos)
    tttr.append(tramos)
    brrr.append(bordes)
    try:
        curv,spline = spl.pegar_fibra(tramos,bordes,window=17,s=10)
        splines.append(spline)        
        curvs.append(curv)
        del(curv,spline)
    except UnboundLocalError:
        splines.append('Nan')
        curvs.append('Nan')
        print('\n',i)
    del(im,sss,fibrass,bbs,fibra,bb,tramos,bordes)
t2 = time()
print(f'\nTarda {t2-t1} segundos en crear y analizar {n_fib} imagenes.')    
#%%
ff = 59
#5 14 15 111 193 214 248 251 285 313 383 404 500 521 571 703 733 782 960
#sacar: 
#revisar: 782
#maso: 5,14,15,111,193,214,248,251,285,313,383,404,500,521,571,703,733,960

# dif en curvatura: 471 641 665 855 

t_spl = np.linspace(0, 1, 10000)
xf, yf = splev(t_spl, splines[ff])
yo, xo = splev(t_spl, splineso[ff])
xf,yf,z = uQuery([xf,yf],u,steps).T
xo,yo,z = uQuery([xo,yo],u,steps).T
curvo = gf.curva(xo,yo)
curvf = gf.curva(xf,yf)
print(curvo,curvf)

plt.figure()
plt.set_cmap('gray')
plt.imshow(imagenes[ff])
plt.imshow(fibras[ff],cmap='gray_r')
plt.plot(xf, yf, 'r-')
plt.plot(xo[::1], yo[::1], 'g-')
#plt.plot(xf-xo[::1], 'r-')
#plt.plot(yf-yo[::1], 'g-')
plt.show()
#%%
#Ahora evaluamos los splines originales y los obtenidos en un mismo array de puntos, para poder comparar las distancias entre los x y lo y y las curvaturas. Siempre comparamos original vs recuperado.
t1 = time()
u = np.linspace(0,1,1000)
steps = 50000
dx, dy, curv = [], [], []
cf_max = []
t_spl = np.linspace(0,1,10000)
for i in range(len(fibras)):
#    if i in [782]: continue
#    if i in [5,14,15,111,193,214,248,251,285,313,383,404,500,521,571,703,733,960]: continue #estan bien, los saco para ver
#    print(i)
    xf,yf = splev(t_spl,splines[i])
    yo,xo = splev(t_spl,splineso[i])
    xf,yf,z = uQuery([xf,yf],u,steps).T
    xo,yo,z = uQuery([xo,yo],u,steps).T
    if np.max(np.abs(xf-xo)) > 20 or np.max(np.abs(yf-yo)) > 20:
        xo = xo[::-1]
        yo = yo[::-1]
    curvf = curvatura_pap(xf,yf)
    curvo = curvatura_pap(xo,yo)
    cf_max.append(np.max(curvf))
#    if np.abs(np.max(curvf-curvo)) >= 0.3:
#        print(i,end=' ')
    minn = 10
    if np.max(np.abs(yf-yo)) > minn or np.max(np.abs(xf-xo)) > minn: print(i,end=' ')
    dx = dx + list(xf-xo)
    dy = dy + list(yf-yo)
    curv = curv+ list((curvf-curvo)/np.abs(curvo+curvf))
    del(xf,xo,yo,yf,z)
t2 = time()
print(f'\nTarda {t2-t1} segundos en evaluar los splines originales y recuperados de {n_fib} imagenes.')

#%%
#Hacemos los histogramas
plt.figure()
plt.hist(dx, bins=100, color='blue', label='x', alpha=0.5, density=True, stacked=True)
plt.hist(dy, bins=100, color='red', label='y', alpha=0.5, density=True, stacked=True)

#x = np.linspace(-4,4,1000)
#gaus = np.exp( -1/(2*0.7209708359440692**2) * (x+0.29839801285882667)**2 ) * 0.55
#plt.plot(x,gaus,'-.')
plt.legend()
plt.show()

plt.figure()
plt.hist(curv,bins=500,density=True, stacked=True)
plt.title('curvatura')
x = np.linspace(-1,1,1000)
gaus = np.exp( -1/(2*0.14101445821084832**2) * (x+0.011881085807854973)**2 ) * 8
plt.plot(x,gaus,'-.')
plt.show()


plt.figure()
plt.hist(np.abs(curv),bins=500,density=True, stacked=True)
plt.title('curvatura')
plt.show()

print(f'Para la diferencia en x, la media es de {np.mean(dx)} y el desvío {np.std(dx)}.')
print(f'Para la diferencia en y, la media es de {np.mean(dy)} y el desvío {np.std(dy)}.')
print(f'Para la diferencia en la curvatura, la media es de {np.mean(curv)} y el desvío {np.std(curv)}.')





