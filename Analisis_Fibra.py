import numpy as np
import matplotlib.pyplot as plt
import Func_Splines as spl
from skimage.io import imread
from scipy.interpolate import splev
from time import time
#from plantcv import plantcv as pcv
#import h5py 
plt.ion()
#%%
from skimage.filters import gabor, gaussian, frangi
from skimage.exposure import adjust_log
from skimage.morphology import thin, skeletonize
from skimage.measure import label, regionprops
#%%
from sklearn.neighbors import NearestNeighbors
import networkx as nx

def ordenar_tramo_rap(tramo,bor,n_nei):
    ii = []
    for i in range(len(bor[0])):
        ind = np.where((tramo[:,0] == bor[i][0]) & (tramo[:,1] == bor[i][1]))
        if len(ind[0]) != 0: ii.append(int(ind[0]))
    pp = (list(tramo[ii[0]]), list(tramo[0]))
    tramo[0], tramo[ii[0]] = pp
    clf = NearestNeighbors(n_neighbors=n_nei).fit(tramo)
    G = clf.kneighbors_graph()
    T = nx.from_scipy_sparse_matrix(G)
    
    order = list(nx.dfs_preorder_nodes(T, 0))
    
    tra_ord = tramo[order]
    return(tra_ord)

def ordenar_fibra(tramos, bor, n_nei):
    fibra = []
    for i in range(len(tramos)):
        tramo = tramos[i]
        if len(tramo) > 2:
            tramo = ordenar_tramo_rap(tramo, bor, n_nei)
        fibra.append(tramo)
    return fibra

def encontrar_fibra(im,bb):
    imr = im[bb[0]:bb[1],bb[2]:bb[3]]
    a = gabor(imr,0.1)
    b = imr - 2*(a[0]-np.min(a[0]) + 10)
    c = gaussian(b,0.7)
    lp = adjust_log(c,1) - 9
    lph = -lp*(lp<0)
    
    lpm = lph[:,:]
    lpme, lpst = np.mean(lph),np.std(lph)
    lpm[lph < lpme-lpst] = np.mean(lph)
    fgg = gaussian(lpm,3)
    lkj = (fgg -gaussian(fgg,30))
    
    rfm = frangi(-lkj)
    rfg = rfm/np.max(rfm)
    muu,stt = np.mean(rfg), np.std(rfg)
    ske = skeletonize(rfg > muu+stt*1.3)
    lal = label(ske)
    props = regionprops(lal)
    for i in range(np.max(lal)):
        arb = props[i].extent
        if arb > 0.05: lal[lal==i+1] = 0
    fib = lal>0
   
    lfib = np.where(fib) 
    yma, xma = np.max(lfib[0]),np.max(lfib[1])
    ymi, xmi = np.min(lfib[0]),np.min(lfib[1])
    
    xmax, ymax = min(1023,xma+bb[2]+1), min(1023,yma+bb[0]+1) 
    xmin, ymin = max(0,xmi+bb[2]), max(0,ymi+bb[0])
    #    bbs = [xmi+bb[2],ymi+bb[0],xma+bb[3],yma+bb[1]]
    #    bbs = [bb[0],bb[2],bb[1],bb[3]]
    bbs = [ymin,xmin,ymax,xmax]
    fii = fib[ymi:yma+1,xmi:xma+1]
    return fii, imr, bbs

def conectar_tramo(tramos,bordes,imprimir=False):
    if len(tramos) > 1 and len(bordes) > 2:
        if imprimir: print('n° tramos:',len(tramos),'\nn° bordes:',len(bordes))
        d0 = 1e6
        for i in range(len(bordes)):
            for j in range(i+1,len(bordes)):
                b1 = np.array(bordes[i])
                b2 = np.array(bordes[j])
                dist = np.sum((b1-b2)**2)
                if dist < d0:
                    n,m = i,j
                    d0 = dist
        conca = tuple([tramos[k] for k in range(len(tramos))])
        tramos = [np.concatenate(conca)]
        bo = np.delete(bordes,[n,m],axis=0)
        nnei = 30
        return tramos, bo, nnei
    else:
        if imprimir: print('n° tramos:',len(tramos),'\nn° bordes:',len(bordes))
        nnei = 2
        bo = bordes
        return tramos, bo, nnei


#%%
num = 2450
ni = '{:04d}'.format(num)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float')
#
#imr = im[0:60,250:550] #800
#imr = im[210:440,570:810] #1800
#imr = im[0:240,380:590] #987
#imr = im[0:280,470:630] #1245
#imr = im[100:400,550:800] #1600
imr = im[400:700,600:830] #2450
#imr = im[850:1024,650:850] #2700

plt.figure()
plt.imshow(im)
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(imr)
plt.colorbar()
plt.show()
#%%
t1 = time()

bb = [400,700,600,830]
fib, imr, bbn = encontrar_fibra(im,bb)

tramos,bordes = spl.cortar_fibra_rap(fib,bbn,cortar_ruido=True)
tramos, bo, nnei = conectar_tramo(tramos,bordes,imprimir=False)
tramos = ordenar_fibra(tramos,bo,nnei)
curv,spline = spl.pegar_fibra(tramos,bo,window=27,s=25)

t2 = time()
print(t2-t1)

t_spl = np.linspace(0, 1, 10000)
xf, yf = splev(t_spl, spline)

plt.figure()
plt.imshow(im)
plt.plot(xf, yf, 'r-')
plt.show()
plt.figure()
plt.imshow(imr)
plt.plot(xf-bb[2], yf-bb[0], 'r-')
plt.show()
plt.figure()
plt.imshow(imr)
plt.show()
#%%
plt.figure()
plt.imshow(fib)
plt.show()

lfib = np.where(fib) 
yma, xma = np.max(lfib[0]),np.max(lfib[1])
ymi, xmi = np.min(lfib[0]),np.min(lfib[1])

xmax, ymax = min(1023,xma+bb[2]+1), min(1023,yma+bb[0]+1) 
xmin, ymin = max(0,xmi+bb[2]), max(0,ymi+bb[0])
#    bbs = [xmi+bb[2],ymi+bb[0],xma+bb[3],yma+bb[1]]
#    bbs = [bb[0],bb[2],bb[1],bb[3]]
bbs = [xmin,ymin,xmax,ymax]
fii = fib[ymi:yma+1,xmi:xma+1]

print(bbs)

plt.figure()
plt.imshow(fii)
plt.show()
np.shape(fii)
#%%