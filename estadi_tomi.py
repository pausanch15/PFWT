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
import scipy.stats as sps
#from itertools import permutations
import h5py 
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
n_fib = 10000
imagenes, fibras = [], []
splines, splineso = [], []
curvs, fibras, tttr, brrr = [],[],[],[]
np.random.seed(12)
t1 = time()
for i in range(n_fib):
#    if i < 4000: continue
    im, sss = gf.genera_im_dinamica(frames=n_int, n_fibras=2, alpha=0.1,N=1000,Nt=8,curvatura=100)
    imagenes.append(im[0])
    splineso.append(sss[0])
    if i%10 == 0: print(i,end=' ')
    fibrass,bbs = spl.encuentra_fibra(im,binariza=70)
    fibra,bb = fibrass[0], bbs[0]
#    fibras.append(fibra)
    tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=False)
    tramos = spl.ordenar_fibra(tramos)
#    tttr.append(tramos)
#    brrr.append(bordes)
    try:
        curv,spline = spl.pegar_fibra(tramos,bordes,window=17,s=10)
        splines.append(spline)
        curvs.append(curv)
        del(curv,spline)
    except UnboundLocalError:
        splines.append('Nan')
        curvs.append('Nan')
        print('\n',i)
#    except IndexError: #para cuando no hay bordes
#        splines.append('Nan')
#        curvs.append('Nan')
#        print('\n',i)
    del(im,sss,fibrass,bbs,fibra,bb,tramos,bordes)
t2 = time()
print(f'\nTarda {t2-t1} segundos en crear y analizar {n_fib} imagenes.')
#%%
# Pruebo de guardarlo en un hdf5
# ver de que no haya un 'Nan' en splines (despues veo que hacer en ese caso)
tf,c1f,c2f,kf = [],[],[],[]
to,c1o,c2o,ko = [],[],[],[]
for i in range(len(splines)):
    if splines[i] != 'Nan':
        tf.append(splines[i][0])
        c1f.append(splines[i][1][0])
        c2f.append(splines[i][1][1])
        kf.append(splines[i][2])
    else:
        tf.append(['Nan'])
        c1f.append(['Nan'])
        c2f.append(['Nan'])
        kf.append(['Nan'])
    
    to.append(splineso[i][0])
    c1o.append(splineso[i][1][0])
    c2o.append(splineso[i][1][1])
    ko.append(splineso[i][2])

with h5py.File('splines.hdf5', 'w') as f:
#    h_im = f.create_group('imagenes')
    h_splf = f.create_group('splines_recons')
    h_splo = f.create_group('splines_orig')
    dt = h5py.special_dtype(vlen=np.dtype('float64'))
    
#    h_im.create_dataset('lista_im',data=imagenes)
    
    h_splf.create_dataset('lista_splf_t',data=tf,dtype=dt)
    h_splf.create_dataset('lista_splf_c1',data=c1f,dtype=dt)
    h_splf.create_dataset('lista_splf_c2',data=c2f,dtype=dt)
    h_splf.create_dataset('lista_splf_k',data=kf)
    
    h_splo.create_dataset('lista_splo_t',data=to)
    h_splo.create_dataset('lista_splo_c1',data=c1o)
    h_splo.create_dataset('lista_splo_c2',data=c2o)
    h_splo.create_dataset('lista_splo_k',data=ko)
    
#with h5py.File('imagenes.hdf5', 'w') as f:
#    h_im = f.create_group('imagenes')
#    h_im.create_dataset('im',data=imagenes)
#%%
imag, splif, splio = [],[],[]
with h5py.File('splines.hdf5', 'r') as f:
#    gim = f.get('imagenes')
#    im_h = gim['lista_im']
    
    gspf = f.get('splines_recons')
    tf_h, kf_h = gspf['lista_splf_t'], gspf['lista_splf_k']
    c1f_h, c2f_h = gspf['lista_splf_c1'], gspf['lista_splf_c2'] 
    
    gspo = f.get('splines_orig')
    to_h, ko_h = gspo['lista_splo_t'], gspo['lista_splo_k']
    c1o_h, c2o_h = gspo['lista_splo_c1'], gspo['lista_splo_c2'] 
    
    for i in range(len(tf_h)):
#        imag.append(im_h[i])
        splif.append([tf_h[i],[c1f_h[i],c2f_h[i]],kf_h[i]])
        splio.append([to_h[i],[c1o_h[i],c2o_h[i]],ko_h[i]])
#%%
ff = 4 #535 616 1006 1361 2045 2089 2421 2510 2807 2811
#+10: 535 616 1006 1361 2045 2089 2421 2510 2807 2811
#+5: 466 640 1220 1324 1586 1788 1881 1937 2163 2507 2553 2604 2662 2755 2831

t_spl = np.linspace(0, 1, 10000)
#xf, yf = splev(t_spl, splif[ff])
#yo, xo = splev(t_spl, splio[ff])
#curv = gf.curva(xo,yo)
#print(curv)

plt.figure()
plt.set_cmap('gray')
plt.imshow(imagenes[-1])
#plt.imshow(fibras[-1],cmap='gray_r')
#plt.plot(xf, yf, 'r-o')
#for i in range(len(tttr[ff])):
#    plt.plot(tttr[ff][i][:,0],tttr[ff][i][:,1],'r.')
#plt.plot(xo[::1], yo[::1], 'g-')
#plt.plot(xf-xo[::1], 'r-')
#plt.plot(yf-yo[::1], 'g-')
plt.show()
#%%
#Ahora evaluamos los splines originales y los obtenidos en un mismo array de puntos, para poder comparar las distancias entre los x y lo y y las curvaturas. Siempre comparamos original vs recuperado.
t1 = time()
u = np.linspace(0,1,1000)
steps = 50000
dx, dy, curv = [], [], []
t_spl = np.linspace(0,1,10000)
for i in range(3000):
    if i%100 == 0: print(i,end=' ')
    if i in [535,616,1006,1361,2045,2089,2421,2510,2807,2811]: continue
    if i in [466,640,1220,1324,1586,1788,1881,1937,2163,2507,2553,2604,2662,2755,2831]: continue 
#    print(i)
    xf,yf = splev(t_spl,splif[i])
    yo,xo = splev(t_spl,splio[i])
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
#    dx.append(np.mean(xf-xo))
#    dy.append(np.mean(yf-yo))
    cmin = 0.5
    cr = (curvf-curvo)/np.abs(curvo+curvf)
    curv = curv+ list(cr)
#    if np.abs(np.max(cr)) > cmin: print(i,end=' ')
    del(xf,xo,yo,yf,z)
t2 = time()
print(f'\nTarda {t2-t1} segundos en evaluar los splines originales y recuperados de {len(splif)} imagenes.')


#%%
dx_m,dx_s = np.mean(dx),np.std(dx)
dy_m,dy_s = np.mean(dy),np.std(dy)
cu_m,cu_s = np.mean(curv),np.std(curv)

dx_scolas = (np.array(dx)-dx_m)/dx_s
dxs = dx_scolas[dx_scolas >-2]
dxs = dxs[dxs<2]
dxs_m, dxs_s = np.mean(dxs), np.std(dxs)

wx = np.linspace(np.min(dxs),np.max(dxs),10000)
wy = np.linspace(np.min(dy),np.max(dy),10000)
#Hacemos los histogramas
plt.figure()
plt.title('Diferencias en x')
gx = np.exp( -1/(2 * dx_s**2) * (wx-dx_m)**2 ) / (dx_s * np.sqrt(2*np.pi)) 
plt.hist((dx-dx_m)/dx_s, bins=150, color='#eb5600', label='x', density=True,edgecolor='black')
#plt.plot(wx,gx,'-')
#plt.legend()
#plt.savefig('histx.png',bbox_inches='tight')
plt.show()
plt.figure()
plt.title('Diferencias en y')
gy = np.exp( -1/(2 * dy_s**2) * (wy-dy_m)**2 ) / (dy_s * np.sqrt(2*np.pi))
plt.hist((dy-dy_m)/dy_s, bins=150, color='#1a9988', label='y',density=True,edgecolor='black')
#plt.plot(wy,gy,'-')
#plt.legend()
#plt.savefig('histy.png',bbox_inches='tight')
plt.show()

print(f'Para la diferencia en x, la media es de {np.mean(dx)} y el desvío {np.std(dx)}.')
print(f'Para la diferencia en y, la media es de {np.mean(dy)} y el desvío {np.std(dy)}.')
#%%
#plt.figure()
#r = np.sqrt(np.array(dx)**2 + np.array(dy)**2)
#plt.hist(r,bins=150)
#plt.show()


#x = np.linspace(-4,4,1000)
#gaus = np.exp( -1/(2*0.7209708359440692**2) * (x+0.29839801285882667)**2 ) * 0.55
#plt.plot(x,gaus,'-.')
#plt.legend()
#plt.show()

plt.figure()
plt.title('Diferencia en curvatura')
plt.hist(curv,bins=500,density=True,color='#1a9988',edgecolor='black',linewidth=0.2) #, stacked=True)
#plt.title('curvatura')
x = np.linspace(-1,1,1000)
gaus = np.exp( -1/(2*cu_s**2) * (x-cu_m)**2 ) / (cu_s * np.sqrt(2*np.pi))
#plt.plot(x,gaus,'-')
plt.savefig('hist_curvat.png',bbox_inches='tight')
plt.show()


#plt.figure()
#plt.hist(np.abs(curv),bins=500,density=True, stacked=True)
#plt.title('curvatura')
#plt.show()
print(f'Para la diferencia en la curvatura, la media es de {np.mean(curv)} y el desvío {np.std(curv)}.')
#%%
#est_x, pv_x = sps.kstest(dx, 'norm',args=(dx_m,dx_s))
print(sps.kstest(dx, 'norm',args=(dx_m,dx_s)))
#est_y, pv_y = sps.kstest(dy, 'norm',args=(dy_m,dy_s))
print(sps.kstest(dy, 'norm',args=(dy_m,dy_s)))
#print(est_x, pv_x, est_y, pv_y)

#%%
mu,sig = 0 ,10
varl = sps.norm.rvs(mu,sig,size=10000)
plt.figure()
w = np.linspace(np.min(varl),np.max(varl),10000)
g = np.exp( -1/(2 * sig**2) * (w-mu)**2 ) / (sig * np.sqrt(2*np.pi))
plt.hist(varl,bins=50,density=True, stacked=True)
plt.plot(w,g,'-')
plt.show()
print( sps.kstest(varl, 'norm',args=(0,10)) )
#%%
#Para sacar los tiempos
n_int = 1
n_fib = 3000
tg, ta = [],[]
np.random.seed(12)
ti = time()
for i in range(n_fib):
    t1 = time()
    im, sss = gf.genera_im_dinamica(frames=n_int, n_fibras=2, alpha=0.1,N=1000,Nt=8,curvatura=100)
    t2 = time()
    tg.append(t2-t1)
    if i%10 == 0: print(i,end=' ')
    t3 = time()
    fibrass,bbs = spl.encuentra_fibra(im,binariza=70)
    fibra,bb = fibrass[0], bbs[0]
    fibras.append(fibra)
    tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=False)
    tramos = spl.ordenar_fibra(tramos)
    try:
        curv,spline = spl.pegar_fibra(tramos,bordes,window=17,s=10)
        t4 = time()
        ta.append(t4-t3)
        del(curv,spline)
    except UnboundLocalError:
        print('\n',i)
    del(im,sss,fibrass,bbs,fibra,bb,tramos,bordes)
tf = time()
print(f'\nTarda {tf-ti} segundos en crear y analizar {n_fib} imagenes.')
#%%
with h5py.File('tiempo.hdf5', 'w') as f:
    h_tm = f.create_group('tiempos')

    h_tm.create_dataset('creacion',data=tg)
    h_tm.create_dataset('analisis',data=ta)
#%%
tg, ta = [],[]
with h5py.File('tiempo.hdf5', 'r') as f:
    ht = f.get('tiempos')
    htg, hta = ht['creacion'], ht['analisis']
    for i in range(len(htg)):
        tg.append(htg[i])
        ta.append(hta[i])
#%%
#plt.figure()
#plt.hist(tg,bins=50,alpha=0.5,label='tg')
#plt.hist(ta,bins=50,alpha=0.5,label='ta')
#plt.legend()
#plt.show()
#
#print(f'{np.mean(ta)},{np.std(ta)}')
#print(f'{np.mean(tg)},{np.std(tg)}')

aes,ges = [],[]
ast,gst = [],[]
for j in range(1,1001):
    a = [np.cumsum(ta[i:i+j])[-1] for i in range(0,len(ta)-j,j)]
    g = [np.cumsum(tg[i:i+j])[-1] for i in range(0,len(tg)-j,j)]
    aes.append(np.mean(a))
    ges.append(np.mean(g))
    ast.append(np.std(a))
    gst.append(np.std(g))
inde = np.linspace(1,1000,1000)
sla = sps.linregress(inde,y=aes)
slg = sps.linregress(inde,y=ges)
print(sla[0],slg[0])

ms = 3
plt.figure()
plt.plot(inde,aes,'s',markersize=ms,label='a',color='black')
plt.errorbar(inde,aes,yerr=ast,fmt='.',markersize=0,ecolor='grey',elinewidth=0.2)
plt.plot(inde,sla[0]*inde+sla[1],'-',color='#eb5600')
#plt.plot(inde,ges,'o',markersize=ms,label='g')
#plt.errorbar(inde,ges,yerr=gst,fmt='.',markersize=0,ecolor='black',elinewidth=0.2)
#plt.plot(inde,slg[0]*inde+slg[1],'-')
#plt.legend(loc=2)
plt.xlabel('N° Fibras')
plt.ylabel('Tiempo (s)')
plt.grid(True)
#plt.savefig('tiempos_nar.png',bbox_inches='tight')
plt.show()
#%%
from PIL import Image
from skimage.morphology import thin, skeletonize, remove_small_objects
from skimage.measure import label, regionprops
image = Image.open('imagenprueba.png') 

ima = np.asarray(image) #la convierto en numpy array
im = ima[:,:,0]
imc = im[20:-20, 100:800] #me quedo entre las lineas 100 y 800 (quito las turbinas)

fibrass,bbs = spl.encuentra_fibra([imc],binariza=200)
fibra,bb = fibrass[0], bbs[0]
tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=False)
tramoso = spl.ordenar_fibra(tramos)
curv,spline = spl.pegar_fibra(tramoso,bordes,window=17,s=10)

#bordes: 4,27; 1,142
#nudos: 165,73; 177,67; 177,68; 176,68; 178,68
xbs = [27,1]
ybs = [4,142]
xns = [165,177,177,176,178]
yns = [73,67,68,68,68]

t_spl=np.linspace(0,1,10000)
xf, yf = splev(t_spl, spline)
plt.figure()
plt.imshow(imc<200,cmap='gray')
#plt.plot(xbs,ybs,'ro')
#plt.plot(xns,yns,'yo')
plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
plt.savefig('binarizada.png',bbox_inches='tight')
plt.show()
#%%
colori = ['red','green','orange','blue']
plt.figure()
for i in range(len(tramoso)):
    if i == 2: continue
    x,y = tramoso[i][:,0], tramoso[i][:,1]
    plt.plot(x,y,'.',color=colori[i])
    plt.plot(x,y,'-',alpha=0.5,color=colori[i])
plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
plt.gca().invert_yaxis()
#plt.savefig('tramos_utiles.png',bbox_inches='tight')
plt.show()
