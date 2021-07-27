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
from itertools import permutations
plt.ion()
#%%
#Genero las imágenes con ruido de estas fibras. Me fijo cuánto tarda en hacer esto. Genero n imágenes entre cada imagen extremo. Uso e extremos (imágenes intermedias).
n = 1
e = 50
np.random.seed(12)
ti = time()
imagenes, splineso = gf.genera_im_dinamica(frames=n, n_fibras=e, alpha=0.1,N=100,Nt=7)
tf = time()
print(tf-ti)
#%%
#Hago esto para todas las imágenes generadas.
#Interpolo todas las fibras. Me fijo cuánto tarda
ti = time()
fibras, bbs = spl.encuentra_fibra(imagenes,binariza=65)
splines = []
for ff in range(len(fibras)):
    if ff in [121,175,464,467]: 
        splines.append('Nan')
        continue
    print(ff,end=' ')
    fibra,bb = fibras[ff], bbs[ff]
    tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=True)
    tramos = spl.ordenar_fibra(tramos)
    curv,spline = spl.pegar_fibra(tramos,bordes,window=17,s=10)
    splines.append(spline)
tf = time()
print(tf-ti)
#%%
def max_curv(splin):
    t_spl = np.linspace(0,1,100000)
#    spline = splev(t_spl,spl)
    spline_1d = splev(t_spl,splin,der=1)
    spline_2d = splev(t_spl,splin,der=2)
    curv = np.abs(spline_2d[0] * spline_1d[1] - spline_1d[0] * spline_2d[1]) / ((spline_1d[0])**2 + (spline_1d[1])**2)**(3/2)
    return np.max(curv) # t_spl, spline
#%%
n_int = 1
n_fib = 50
imagenes, fibras = [], []
splines, splineso = [], []
curvs, fibras, tttr, brrr = [],[],[],[]
np.random.seed(12)
t1 = time()
for i in range(n_fib):
    mcu = 10
    repe = 0
    while (mcu > 1.5) and (repe < 10):
        im, sss = gf.genera_im_dinamica(frames=n_int, n_fibras=2, alpha=0.1,N=1000,Nt=8)
        mcu = max_curv(sss[0])
        repe += 1
    imagenes.append(im[0])
    splineso.append(sss[0])
#    if i ==26: continue
    print(i,end=' ')
    fibrass,bbs = spl.encuentra_fibra(im,binariza=107)
    fibra,bb = fibrass[0], bbs[0]
    fibras.append(fibra)
    tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=True)
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
print(t2-t1)
#%%
ff = 26 #9,15,26

t_spl = np.linspace(0, 1, 10000)
xf, yf = splev(t_spl, splines[ff])
yo, xo = splev(t_spl, splineso[ff])
curv = gf.curva(xo,yo)
print(curv)

u = np.linspace(0,1,1001)
steps = 50000 # The more subdivisions the better
#t1 = time()
#xf,yf,z = uQuery([xf,yf],u,steps).T
#xo,yo,z = uQuery([xo,yo],u,steps).T
#t2 = time()
#print(t2-t1)


plt.figure()
plt.set_cmap('gray')
plt.imshow(imagenes[ff])
plt.imshow(fibras[ff],cmap='gray_r')
plt.plot(xf, yf, 'r-')
plt.plot(xo[::1], yo[::1], 'g-')
#plt.plot(xf-xo[::1], 'r-')
#plt.plot(yf-yo[::1], 'g-')
plt.show()
#plt.figure()
#plt.plot(curvs[ff])
#print(np.max(curvs[ff]))
#plt.show()
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
t1 = time()
u = np.linspace(0,1,1001)
steps = 50000
dx, dy, dxdy = [], [], []
t_spl = np.linspace(0,1,10000)
for i in range(len(fibras)):
    if i in [15]: continue
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
    minn = 5
    if np.max(np.abs(yf-yo)) > minn or np.max(np.abs(xf-xo)) > minn: print(i,end=' ')
    dx = dx + list(xf-xo)
    dy = dy + list(yf-yo)
    dxdy = dxdy + list((xf-xo)+(yf-yo))
t2 = time()
print(t2-t1)
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
print('dx:',np.mean(dx),np.std(dx), '\ndy:',np.mean(dy),np.std(dy))
#%%
#Agrego otra forma de hacer lo mismo, pero que puede tener errores.
#Armo imagenes, encuentro fribras y hago estadística
# dx, dy, dxdy = [], [], []
dx_f, dy_f, dxdy_f = open('dx_f.txt', "w"), open('dy_f.txt', "w"), open('dxdy_f.txt', "w")

j = 0
while j < 4:
    print(f'Vamos por la tanda {j}.')
    
    n = 10
    e = 50
    np.random.seed(12)
    ti = time()
    imagenes, splineso = gf.genera_im_dinamica(frames=n, n_fibras=e, drift=0, alpha=0.1,N=10,Nt=7)
    tf = time()
    print(f'Tarda {tf-ti} segundos en generar {len(imagenes)} imágenes.')

    ti = time()
    fibras, bbs = spl.encuentra_fibra(imagenes,binariza=63)
    splines = []
    for ff in range(len(fibras)):
        # print(ff,end=' ')
        fibra,bb = fibras[ff], bbs[ff]
        tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=False)
        tramos = spl.ordenar_fibra(tramos)
        curv,spline = spl.pegar_fibra(tramos,bordes,window=21,s=10)
        splines.append(spline)
    tf = time()
    print(f'\nTarda {tf-ti} segundos en interpolar todas las fibras de las imágenes.')

    u = np.linspace(0,1,1001)
    steps = 50000
    dx, dy, dxdy = [], [], []
    t_spl = np.linspace(0,1,10000)
    for i in range(len(fibras)):
        if i in [7,32,37]: continue
        xf,yf = splev(t_spl,splines[i])
        yo,xo = splev(t_spl,splineso[i])
        xf,yf,z = uQuery([xf,yf],u,steps).T
        xo,yo,z = uQuery([xo,yo],u,steps).T
        if np.max(np.abs(xf-xo)) > 20 or np.max(np.abs(yf-yo)) > 20:
            xo = xo[::-1]
            yo = yo[::-1]
        if np.max(np.abs(yf-yo)) > 20 or np.max(np.abs(xf-xo)) > 4: print(i,end=' ')
        dx = dx + list(xf-xo)
        dy = dy + list(yf-yo)
        dxdy = dxdy + list((xf-xo)+(yf-yo))

    np.savetxt(dx_f, dx, delimiter=",")
    np.savetxt(dy_f, dy, delimiter=",")
    np.savetxt(dxdy_f, dxdy, delimiter=",")

    j += 1

dx_f.close()
dy_f.close()
dxdy_f.close()

#Hago los hisotgramas
dx_f, dy_f, dxdy_f = np.loadtxt('dx_f.txt', delimiter=','), np.loadtxt('dy_f.txt', delimiter=','), np.loadtxt('dxdy_f.txt', delimiter=',')

ti = time()
histx, limx = np.histogram(dx_f, bins='auto')
histy, limy = np.histogram(dy_f, bins='auto')
histxy, limxy = np.histogram(dxdy_f, bins='auto')
tf = time()
print(f'Tarda {tf-ti} segundos en generar el histograma para {len(imagenes)} imágenes.')

ti = time()
plt.figure()
plt.plot(histx, color='blue', label='x', alpha=0.5)
plt.plot(histy, color='red', label='y', alpha=0.5)
plt.legend()
plt.show()
tf = time()
print(f'Tarda {tf-ti} segundos hacer el gráfico de los histogramas')
