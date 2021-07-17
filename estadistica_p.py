#Pruebo hacer estadística con muchas imágenes.
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
plt.ion()
#%%
#Genero las imágenes con ruido de estas fibras. Me fijo cuánto tarda en hacer esto. Genero n imágenes entre cada imagen extremo. Uso e extremos (imágenes intermedias).
n = 1
e = 50
np.random.seed(12)
ti = time()
imagenes, splineso = gf.genera_im_dinamica(frames=n, n_fibras=e, drift=0, alpha=0.1,N=10,Nt=7)
tf = time()
print(f'Tarda {tf-ti} segundos en generar {len(imagenes)} imágenes.')
#%%
#Hago esto para todas las imágenes generadas.
#Interpolo todas las fibras. Me fijo cuánto tarda
ti = time()
fibras, bbs = spl.encuentra_fibra(imagenes,binariza=63)
splines = []
for ff in range(len(fibras)):
    print(ff,end=' ')
    fibra,bb = fibras[ff], bbs[ff]
    tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=False)
    tramos = spl.ordenar_fibra(tramos)
    curv,spline = spl.pegar_fibra(tramos,bordes,window=21,s=10)
    splines.append(spline)
tf = time()
print(f'\nTarda {tf-ti} segundos en interpolar todas las fibras de las imágenes.')
#%%
ff = 23 #23,29 #5,41

t_spl = np.linspace(0, 1, 10000)
xf, yf = splev(t_spl, splines[ff])
yo, xo = splev(t_spl, splineso[ff])

u = np.linspace(0,1,1001)
steps = 50000 # The more subdivisions the better
t1 = time()
xf,yf,z = uQuery([xf,yf],u,steps).T
xo,yo,z = uQuery([xo,yo],u,steps).T
t2 = time()
print(t2-t1)

#np.interp()

plt.figure()
plt.set_cmap('gray')
plt.imshow(imagenes[ff])
#plt.imshow(fibras[ff],cmap='gray_r')
ini,fin = 0,-1
plt.plot(xf[ini:fin], yf[ini:fin], 'r-')
plt.plot(xo[::-1][ini:fin], yo[::-1][ini:fin], 'g-')
#plt.plot(xf-xo[::1], 'g-')
#plt.plot(yf-yo[::1], 'g--')
#plt.plot(xfn-xon[::1], 'r-')
#plt.plot(yfn-yon[::1], 'r--')
#plt.plot(xfn,yfn,'r.')
#plt.plot(xon,yon,'g.')
plt.show()
#%%
df = np.sqrt((xf[:-1]-xf[1:])**2 + (yf[:-1]-yf[1:])**2)
do = np.sqrt((xo[:-1]-xo[1:])**2 + (yo[:-1]-yo[1:])**2)

print(np.max(xf-xo[::-1]),np.max(xf-xo[::-1]))     

plt.figure()
plt.plot(df,'r.')
plt.plot(do,'g.')
plt.show()
#%%

def equies(spl,N=100): #lo que intente para equiespaciar, en general funciona pero no es muy bueno
    t_spl = np.linspace(0, 1, N*100)
    x, y = splev(t_spl, spl)
    d = np.sqrt((x[:-1]-x[1:])**2 + (y[:-1]-y[1:])**2)
    lar = np.sum(d)
    dist = lar/N
    xn, yn = [x[0]],[y[0]]
    for i in range(len(t_spl)):
        xx,yy = x[i],y[i] 
        if np.sqrt((xx-xn[-1])**2 + (yy-yn[-1])**2) >= dist:
            xn.append(xx)
            yn.append(yy)
    return np.array(xn),np.array(yn)

def div_cur(x,y):
    xs, ys = [], []
    
    
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
#%%
#Hago el histograma para todas
u = np.linspace(0,1,1001)
steps = 50000
dx, dy, dxdy = [], [], []
t_spl = np.linspace(0,1,10000)
for i in range(len(fibras)):
    if i in [7,32,37]: continue
    xf,yf = splev(t_spl,splines[i])
    yo,xo = splev(t_spl,splineso[i])
#    xf,yf = equies(splines[i],N=100)
#    yo,xo = equies(splineso[i],N=100)
#    if len(xf)!= len(xo): print(i)
    xf,yf,z = uQuery([xf,yf],u,steps).T
    xo,yo,z = uQuery([xo,yo],u,steps).T
    if np.max(np.abs(xf-xo)) > 20 or np.max(np.abs(yf-yo)) > 20:
        xo = xo[::-1]
        yo = yo[::-1]
    if np.max(np.abs(yf-yo)) > 20 or np.max(np.abs(xf-xo)) > 4: print(i,end=' ')
    dx = dx + list(xf-xo)
    dy = dy + list(yf-yo)
    dxdy = dxdy + list((xf-xo)+(yf-yo))
#%%   
plt.figure()
#plt.hist(dx, bins='auto', color='blue', label='x', alpha=0.5, density=True, stacked=True)
#plt.hist(dy, bins='auto', color='red', label='y', alpha=0.5, density=True, stacked=True)
plt.hist(dx, bins=50, color='blue', label='x', alpha=0.5, density=True, stacked=True)
plt.hist(dy, bins=50, color='red', label='y', alpha=0.5, density=True, stacked=True)
plt.legend()
plt.show()
plt.figure()
#plt.hist(dxdy,bins='auto')
plt.hist(dxdy,bins=50)
plt.title('(xf-xo)+(yf-yo)')
plt.show()

