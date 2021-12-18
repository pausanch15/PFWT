import numpy as np
import matplotlib.pyplot as plt
import Func_Splines as spl
from skimage.io import imread
from scipy.interpolate import splev
from time import time
from plantcv import plantcv as pcv
#import h5py 
plt.ion()
#%%
from skimage.filters import gabor, gaussian, frangi
from skimage.exposure import adjust_log
from skimage.morphology import thin, skeletonize, disk, binary_closing
from skimage.filters.rank import enhance_contrast
from skimage.util import img_as_ubyte
from skimage.measure import label, regionprops
#%%
from sklearn.neighbors import NearestNeighbors
import networkx as nx

def ordenar_tramo_rap(tramo,bor,n_nei):
    ii = []
    for i in range(len(bor[:,0])):
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

def encontrar_fibra(im,bb,prun=50,std_mul=1.3):
    imr = im[bb[0]:bb[1],bb[2]:bb[3]]
    a = gabor(imr,0.1)
    b = imr - 2*(a[0]-np.min(a[0]) + 10)
    c = gaussian(b,0.7,mode='mirror')
    lp = adjust_log(c,1) - 9
    lph = -lp*(lp<0)
    
    lpm = lph[:,:]
    lpme, lpst = np.mean(lph),np.std(lph)
    lpm[lph < lpme-lpst] = np.mean(lph)
    fgg = gaussian(lpm,3,mode='mirror')
    lkj = (fgg -gaussian(fgg,30,mode='mirror'))
    
    rfm = frangi(-lkj)
    rfg = rfm/np.max(rfm)
    muu,stt = np.mean(rfg), np.std(rfg)
    ske = skeletonize(rfg > muu+stt*std_mul)
    lal = label(ske)
    props = regionprops(lal)
    for i in range(np.max(lal)):
        arb = props[i].orientation
        arre = props[i].extent
#        print(i+1,'\t',arb,'\t',arre)
        if (arre > 0.21) or (arb > -0.4): lal[lal==i+1] = 0
#    fib = lal>0
    fre = pcv.morphology.skeletonize(lal>0)
    fib, si,so = pcv.morphology.prune(fre,size=prun)
   
    lfib = np.where(fib) 
    yma, xma = np.max(lfib[0]),np.max(lfib[1])
    ymi, xmi = np.min(lfib[0]),np.min(lfib[1])
    
    xmax, ymax = min(1023,xma+bb[2]+1), min(1023,yma+bb[0]+1) 
    xmin, ymin = max(0,xmi+bb[2]), max(0,ymi+bb[0])
    #    bbs = [xmi+bb[2],ymi+bb[0],xma+bb[3],yma+bb[1]]
    #    bbs = [bb[0],bb[2],bb[1],bb[3]]
    bbs = [ymin,xmin,ymax,xmax]
    fii = fib[ymi:yma+1,xmi:xma+1]
    return fii>0, imr, bbs

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
num = 1700
ni = '{:04d}'.format(num)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float')
#
#imr = im[0:100,250:550] #800
#imr = im[210:440,570:810] #1800
#imr = im[0:240,380:590] #987
#imr = im[0:280,470:630] #1245
#imr = im[100:400,550:800] #1600
#imr = im[400:700,600:830] #2450
#imr = im[850:1024,650:850] #2700
#imr = im[0:70,220:550] #819
#imr = im[0:100, 220:510] # 750
imr = im[220:470,550:800]

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

bb = [3,285,477,668]
fib, imr, bbn = encontrar_fibra(im,bb,std_mul=1.5,prun=50)

tramos,bordes = spl.cortar_fibra_rap(fib,bbn,cortar_ruido=False)
tr, bo, nnei = conectar_tramo(tramos,bordes,imprimir=False)
#
#plt.figure()
#for t in tramos:
#    plt.plot(t[:,0],t[:,1],'.')
#plt.plot(bordes[:,0],bordes[:,1],'k.')
#plt.plot(bo[:,0],bo[:,1],'r.')
#plt.show()

tramos = spl.ordenar_fibra(tramos)
curv,spline = spl.pegar_fibra(tramos,bo,window=27,s=25)
#
t2 = time()
print(t2-t1)
#
t_spl = np.linspace(0, 1, 10000)
xf, yf = splev(t_spl, spline)
#
plt.figure()
plt.imshow(im)
plt.plot(xf, yf, 'r-')
plt.show()
plt.figure()
plt.imshow(imr)
plt.plot(xf-bb[2], yf-bb[0], 'r-')
plt.show()
plt.figure()
plt.imshow(fib)
plt.show()
#%%
nb1, nb2, = max(0,bbn[0]-20), min(1024,bbn[2]+20)
nb3, nb4 = max(0,bbn[1]-20), min(1024,bbn[3]+20)
bbs = [nb1,nb2,nb3,nb4]

plt.figure()
plt.imshow(im[bb[0]:bb[1],bb[2]:bb[3]])
plt.show()
plt.figure()
plt.imshow(im[bbn[0]:bbn[2],bbn[1]:bbn[3]])
plt.show()
plt.figure()
plt.imshow(im[bbs[0]:bbs[1],bbs[2]:bbs[3]])
plt.show()
#%%
#==============================================================================
# Pruebo correr varias fibras
#==============================================================================
t1 = time()
splines = []
#bb = [0,100,220,510] #para 760
bb = [0,268,418,659] #para 1100
#bb = [80,370,550,750] #para 1500
#bb = [220,470,550,800] #para 1700
bbt = [bb]
for num in range(1100,1400):
    ni = '{:04d}'.format(num)
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
    #ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
    im = np.array(ima,dtype='float')
    
    if num%30 == 0: print(num,end=' ')
    
    try:
        stdmul, disc, mmax, tama, tti = 1.7, 10, 0.8, 145, 'wrap'
        if num in [899,900]: mmax=1.8

        if num in range(1175,1180): mmax = 1.5 
#        if num in range(1190,1312): stdmul=1.2
#        if num in [1308,1309]: mmax = 0
#        if num in range(1312,1368): stdmul=1
#        if num == 142: disc,mmax = 15,1.1
#        if num == 143: 
#            splines.append(spline)
#            continue
#        if num in range(1368,1395): stdmul,disc=1, 15
#        if num in range(1395,1500): stdmul,disc=0.9, 15
#        if num in range(1368,1500): stdmul, disc= 1.05, 15

#        if num >= 1500: mmax,stdmul,disc,tti,tama = -0.1,0.5,11,'mirror',90
#        if num == 1565: mmax = 2
#        if num >=1589: stdmul = 0.7
        
        fib,imr,bbn = encontrar_fibra2(im,bb,std_mul=stdmul,prun=20,disco=disc,mulm=mmax,tama=tama,tipog=tti)
#        fib,imr,bbn = encontrar_fibra(im,bb,prun=50,std_mul=1.3)
        tramos,bordes = spl.cortar_fibra_rap(fib,bbn,cortar_ruido=True)
        tr, bo, nnei = conectar_tramo(tramos,bordes,imprimir=False)
        tramos = spl.ordenar_fibra(tramos)
        curv,spline = spl.pegar_fibra(tramos,bo,window=27,s=25)
    except:
        print('Fallo',num)
        break
    
    splines.append(spline)
    
    nb1, nb2, = max(0,bbn[0]-40), min(1024,bbn[2]+40)
    nb3, nb4 = max(0,bbn[1]-40), min(1024,bbn[3]+40)
    bb = [nb1,nb2,nb3,nb4]
    bbt.append(bb)

t2 = time()
print(t2-t1)
#%%
n = -1100 + 1325
print(bbt[n])
ni = '{:04d}'.format(1100+n)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float')

    
t_spl = np.linspace(0,1,100000)
spline = splev(t_spl,spl)

plt.figure()
plt.imshow(im)
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(im)
plt.plot(xf, yf, 'r-')
plt.show()
#plt.figure()
#plt.imshow(fib)
##plt.colorbar()
#plt.show()
#%%
t_spl = np.linspace(0, 1, 1000)
nn = len(splines)
largos = []
plt.figure()
for n in range(nn):
    xf, yf = splev(t_spl, splines[n])
#    plt.imshow(np.zeros_like(im))
    plt.plot(xf-np.mean(xf),-(yf-np.mean(yf)))
    lar = largo_fib(xf,yf)
    largos.append(lar)
plt.show()
#%%
plt.figure()
plt.plot(largos,'-o')
plt.grid()
plt.show()
#%%
def largo_fib(xf,yf):
    nf = len(xf)
    td = 0
    for i in range(1,nf):
        di = np.sqrt( (xf[i]-xf[i-1])**2 + (yf[i]-yf[i-1])**2  )
        td += di
    return td
#%%
#==============================================================================
# Pruebo fibras que fallen
#==============================================================================
num = 800
#num = 1308
#num = 1100+208 #1100+75 #64,98,426,427

ni = '{:04d}'.format(num)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float') 


plt.figure()
plt.imshow(im)
plt.show()
#%%
#bb = [0,126,227,531] # 819
#bb = [400,700,600,830] #2450
#bb = [0,280,450,650] #1242
#bb = [210,440,570,810] #1800 
#bb = [0, 300, 500, 650] #1308
bb = [0,70,250,520]
#bb = bbt[208]
#bb = bbt[1242-1100]

imr = im[bb[0]:bb[1],bb[2]-40:bb[3]+40]
#a1 = imr - gaussian(imr,20)
#a = gabor(a1,0.1,mode='mirror')
#b = a1 - 2*(a[0]-np.min(a[0]) + 10)
#c = gaussian(b,0.7,mode='wrap')
#lp = adjust_log(c-np.min(c),1) - 9
#itn = lp #- gaussian(lp,30)
#lph = -itn*(itn<0)
#
#lpm = lph[:,:]
#lpme, lpst = np.mean(lph),np.std(lph)
#lpm[lph < lpme-lpst] = np.mean(lph)
#fgg = gaussian(lpm,3,mode='mirror')
#lkj = (fgg -gaussian(fgg,30,mode='mirror'))
#
#rfm = frangi(-lkj)
#rfg = rfm/np.max(rfm)

from scipy.signal import hilbert, find_peaks
from scipy.interpolate import CubicSpline, splev, splrep, splprep


#todo = np.zeros_like(imr)
#for i in range(bb[1]-bb[0] ):
#    lin = ed[i,:] - np.mean(ed[i,:])
#    x = np.arange(0,1024,1)
#    a, _ = find_peaks(-lin,distance=10)
#    
#    spl = splrep(x[a],lin[a])
#    xf = np.arange(0,imr.shape[1])
#    yf = splev(xf,spl )
#    todo[i,:] = yf

#qsc = b - gaussian(b,50,mode='mirror')
#ed = -sato(qsc)

a1 = imr - gaussian(imr,20)
a7 = gabor(a1,0.7)
a = gabor(a1,0.1)
b = a1 - 2*(a[0]-np.min(a[0]) + 10)
c = b - gaussian(b,30,mode='mirror')
d = gaussian(c,0.5)

th = gabor(d*(d<0),0.7)
ttt = th[1]+th[0]

ggg = gaussian(ttt[:,40:-40],3) - gaussian(ttt[:,40:-40],5)
hi = frangi(ggg)
hii = hi / np.max(hi)
mii = minimum(hi,np.array([[0,0,0],
                           [1,1,1],
                           [0,0,0]]))
rso = remove_small_objects(mii>0,min_size=20)
rbr = skeletonize(dilation(rso,disk(3)))

#asw = a7[0] - a[0]
##resy = asw * (asw<0)
#resy = (asw-np.mean(asw)) / np.max(np.abs((asw-np.mean(asw))))
##doo = gaussian(resy,3)-gaussian(resy,5)


plt.figure()
plt.imshow( ttt )
plt.colorbar()
plt.show()
#plt.figure()
#plt.imshow( a[0] )
#plt.colorbar()
#plt.clim(-50,150)
#plt.show()

plt.figure()
plt.imshow( mii )
#plt.colorbar()
plt.show()

#plt.figure()
#plt.hist(hii.flatten(), bins=50)
#plt.show()
#%%
ghj = enhance_contrast(img_as_ubyte(rfg),disk(11))
mme, sst = np.mean(ghj), np.std(ghj)
print(mme, sst)
stmul = 1
kse = (ghj > mme+sst*stmul)
kse[:,:40] =  np.zeros_like(kse[:,:40])
kse[:,-40:] =  np.zeros_like(kse[:,-40:])
lal = label( kse )
props = regionprops(lal)

plt.figure()
plt.imshow(ghj)
plt.show()
plt.figure()
plt.imshow( kse )
plt.show()
plt.figure()
plt.imshow(lal)
plt.show()
plt.figure()
plt.hist((ghj[ghj>0]).flatten(),bins=30)
plt.show()
#%%
ertt = kse * rfg

plt.figure()
plt.imshow( ertt )
plt.colorbar()
plt.show()
#%%

#tyh = kse*ghj
#print( np.mean(tyh[tyh>0]), np.std(tyh[tyh>0]), np.median(tyh[tyh>0]) )
#mmaxx = np.mean(tyh[tyh>0]) - np.std(tyh[tyh>0]) * 1
#for i in range(np.max(lal)):
#    arb = props[i].area
#    ar = props[i].orientation
##    are = (arb[2]-arb[0]) * (arb[3]-arb[1]) 
##    arre = props[i].intensity_max
#    arre = np.max(tyh[lal==i+1])
#    print(i+1,'\t',ar,'\t',arre)
##    if arre > 0.21 or np.abs(ar)<0.1: lal[lal==i+1] = 0
##    if arb<100 or np.abs(ar)<0.1: lal[lal==i+1] = 0
#    if not (arb > 0 and arre > 0 and np.abs(ar)>0): lal[lal==i+1] = 0
#fre = pcv.morphology.skeletonize(lal>0)
#fib, si,so = pcv.morphology.prune(fre,size=20)

#fib = skeletonize(dilation(lal>0,disk(15)))
#fib = skeletonize(binary_closing(lal>0,disk(50)))

plt.figure()
plt.imshow( fib )
plt.show()
#%%
tramos,bordes = spl.cortar_fibra_rap(fib>0,bb,cortar_ruido=False)
tr, bo, nnei = conectar_tramo(tramos,bordes,imprimir=False)
tramos = spl.ordenar_fibra(tramos)
curv,spline = spl.pegar_fibra(tramos,bo,window=27,s=25)

t_spl = np.linspace(0, 1, 10000)
xf, yf = splev(t_spl, spline)

#plt.figure()
#plt.imshow(im)
#plt.colorbar()
#plt.show()
plt.figure()
plt.imshow(im)
plt.plot(xf, yf, 'r-')
plt.show()

#%%
def encontrar_fibra2(im,bb,prun=20,std_mul=2,disco=7,mulm=0.8,tama=145,tipog='wrap'):
    imr = im[bb[0]:bb[1],bb[2]:bb[3]]
    a = gabor(imr,0.1)
    b = imr - 2*(a[0]-np.min(a[0]) + 10)
    c = gaussian(b,0.7,mode=tipog)
    lp = adjust_log(c,1) - 9
    lph = -lp*(lp<0)
    
    lpm = lph[:,:]
    lpme, lpst = np.mean(lph),np.std(lph)
    lpm[lph < lpme-lpst] = np.mean(lph)
    fgg = gaussian(lpm,3,mode=tipog)
    lkj = (fgg -gaussian(fgg,30,mode=tipog))
    

    rfm = frangi(-lkj)
    rfg = rfm/np.max(rfm)

    ghj = enhance_contrast( img_as_ubyte(rfg) ,disk(disco))
    mme, sst = np.mean(ghj), np.std(ghj)
    
#    kse = skeletonize(ghj > mme+sst*std_mul)
    kse = (ghj > mme+sst*std_mul)
    kse[:,:5] =  np.zeros_like(kse[:,:5])
    kse[:,-5:] =  np.zeros_like(kse[:,-5:])
    tyh = kse*ghj
    mmaxx = np.mean(tyh[tyh>0]) - np.std(tyh[tyh>0]) * mulm
    lal = label( kse )
    props = regionprops(lal)
    for i in range(np.max(lal)):
#        arre = props[i].extent
        ori = props[i].orientation
        arb = props[i].area
        mma = np.max(tyh[lal==i+1])
#        print(i+1,'\t',arre,'\t',arb)
#        if arre > 0.21 or np.abs(ar)<0.2 : lal[lal==i+1] = 0
#        if arb < 150 or np.abs(ori)<0.1 or arre>0.67: lal[lal==i+1] = 0
        if not (arb > tama and mma > mmaxx and np.abs(ori)>0.07): lal[lal==i+1] = 0

#    fib = lal>0
    fre = pcv.morphology.skeletonize(lal>0)
    fib, si,so = pcv.morphology.prune(fre,size=prun)
   
    lfib = np.where(fib) 
    yma, xma = np.max(lfib[0]),np.max(lfib[1])
    ymi, xmi = np.min(lfib[0]),np.min(lfib[1])
    
    xmax, ymax = min(1023,xma+bb[2]+1), min(1023,yma+bb[0]+1) 
    xmin, ymin = max(0,xmi+bb[2]), max(0,ymi+bb[0])
    #    bbs = [xmi+bb[2],ymi+bb[0],xma+bb[3],yma+bb[1]]
    #    bbs = [bb[0],bb[2],bb[1],bb[3]]
    bbs = [ymin,xmin,ymax,xmax]
    fii = fib[ymi:yma+1,xmi:xma+1]
    return fii>0, imr, bbs
#%%
