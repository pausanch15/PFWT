import numpy as np
import matplotlib.pyplot as plt
import Func_Splines as spl
import Continuacion_Fourier as cf
from skimage.io import imread
from scipy.interpolate import splev
from plantcv import plantcv as pcv
import h5py 
plt.ion()
#%%
# traigo imagenes
gris = np.zeros((1024,1024))
for i in range(1,16):
    ni = '{:04d}'.format(i)
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0003\ID_0_C1S000300'+ni+'.tif')
#    ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0006/ID_0_C1S000600'+ni+'.tif')
    ima = np.array(ima,dtype='float')
    gris += ima
gris = gris/15

ref = np.zeros((1024,1024))
for i in range(1,14):
    ni = '{:04d}'.format(i)
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0002\ID_0_C1S000300'+ni+'.tif')
#    ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
    ima = np.array(ima,dtype='float')
    ref += ima
ref = ref/13
#%%
plt.figure()
plt.imshow(gris,cmap='gray')
plt.show()
plt.figure()
plt.imshow(ref,cmap='gray')
plt.show()
plt.figure()
plt.imshow(ref-gris,cmap='gray')
plt.show()
#%%
from scipy.signal import convolve2d, medfilt
from skimage.filters import gaussian, frangi, sato, gabor, laplace, sobel, roberts, hessian, inverse
from skimage.filters.rank import minimum, enhance_contrast, gradient
from skimage.morphology import disk, flood, flood_fill
from skimage.measure import label, regionprops
from skimage.util import img_as_ubyte
from skimage.exposure import adjust_gamma, adjust_log
from skimage.morphology import thin, skeletonize, remove_small_objects, dilation, binary_erosion, binary_closing, closing
#%%
num = 1800
ni = '{:04d}'.format(num)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float')

plt.figure()
plt.imshow(im[210:440,570:810])
plt.colorbar()
plt.show()
#%%
imr = im[210:440,570:810]

a = gabor(imr,0.1) #IMPORTANTE, no pongas el valor menor a 0.1 en esta funciona (te puede comer la memoria)
b = imr - 2*(a[0]-np.min(a[0]) + 10)
#ver = np.array([[0,1,0],
#                [0,1,0],
#                [0,1,0],
#                [0,1,0],
#                [0,1,0]])
#c = minimum(b/np.max(b),disk(4))
c = gaussian(b,0.7)
#d = flood_fill(-c,(84,142),200,tolerance=27)
#d = laplace(c,ksize=3)

d = (c - np.mean(c))
e = d * (d<0)
fg = frangi(-e**5)

plt.figure()
plt.imshow( fg==0 )
#plt.colorbar()
plt.show()
#%%
f = hessian(e)
g = f * (f<0.7) * (f>0.25)
h = remove_small_objects((g[:,5:-5])>0,connectivity=1)
#d[d>75] = 10
#e = frangi(-d) 
#f = remove_small_objects((e[:,5:-5])>0.20,connectivity=1)
#g = binary_closing(f)
#fibra = thin(g)
fibra_r = skeletonize(binary_erosion(dilation(h)))
kernel = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]])
cf = convolve2d(fibra_r,kernel)
convolved_fibra = cf[1:-1,1:-1] * fibra_r
by, bx = np.array(np.where(convolved_fibra == 2))
dis = 1e3
for i in range(len(bx)):
    for j in range(i+1,len(bx)):
        bb1p = np.array([bx[i],by[i]]) 
        bb2p = np.array([bx[j],by[j]])
        dist = np.sqrt( np.sum((bb2p-bb1p)**2) )
        if dist < dis:
            dis = dist
            bb1, bb2 = bb1p, bb2p
bxs = [bb1[0],bb2[0]]
x, slo = np.linspace(np.min(bxs),np.max(bxs),100), (bb2[1]-bb1[1])/(bb2[0]-bb1[0])
lin = slo * (x - bb1[0]) + bb1[1]         
im_fibra, x_ed, y_ed = np.histogram2d(x+236,lin+550, [440-210,805-575], [[210,440],[575,805]] )

klg = (im_fibra + binary_closing(h))>0
fff = binary_closing(klg)

plt.figure()
plt.imshow(skeletonize(fff))
#plt.colorbar()
plt.show()
#plt.figure()
#plt.imshow(im_fibra+convolved_fibra)
#plt.show()
#%%
bb = [210,575,440,810] 

tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=False)
tramos = spl.ordenar_fibra(tramos)
curv,spline = spl.pegar_fibra(tramos,bordes,window=15,s=15)

t_spl = np.linspace(0, 1, 10000)
xf, yf = splev(t_spl, spline)

plt.figure()
plt.imshow(im,cmap='gray')
plt.plot(xf, yf, 'r-')
#plt.colorbar()
plt.show()
plt.figure()
plt.imshow(im,cmap='gray')
#plt.colorbar()
plt.show()
#%%
t1 = time()

imr = im[210:440,570:810]
a = gabor(imr,0.1)
b = imr - 2*(a[0]-np.min(a[0]) + 10)
bm = (b - np.mean(b)) / np.max(b)
d = enhance_contrast(img_as_ubyte(-bm),disk(15))
e = remove_small_objects(d>0,connectivity=5)
f = gaussian(d,0.7)
#g = closing(f,disk(5))
g = remove_small_objects(f>0.04,connectivity=1)
exten = 50
h = np.pad(g[5:-5,5:-5],[exten,exten],mode='constant')
j = binary_closing(h,disk(50))
k = skeletonize(j)

t2 = time()
print(t2-t1)

plt.figure()
plt.imshow(k)
#plt.colorbar()
plt.show()
#%%
fibra = k[45:-45,45:-45]
bb = [210,570,440,810] 

tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=False)
tramos = spl.ordenar_fibra(tramos)
curv,spline = spl.pegar_fibra(tramos,bordes,window=11,s=10)

t_spl = np.linspace(0, 1, 10000)
xf, yf = splev(t_spl, spline)

plt.figure()
plt.imshow(im,cmap='gray')
plt.plot(xf, yf, 'r-')
#plt.colorbar()
plt.show()
plt.figure()
plt.imshow(im,cmap='gray')
#plt.colorbar()
plt.show()

#%%
#%%
num = 800
ni = '{:04d}'.format(num)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float')

imr = im[0:70,250:520]
#para num=1850: [250:460,570:820]
#para num=800: [0:70,250:520]

ims = frangi(imr)
imsb = ims * (ims<0.95) * (ims>0.24)
ver = np.array([[0,0,0],[1,1,1],[0,0,0]])
imm = minimum(imsb,ver)
#immb = imm[:,5:-5]>0
#img = (gaussian(immb,1))>0.05
#imag = binary_erosion(dilation( immb  ))
exten = 15
padi = np.pad(imm[:,5:-5],[exten,exten],mode='constant')
imag = closing(padi,disk(15))
fibra = remove_small_objects( skeletonize(imag>0), connectivity=2)

#plt.figure()
#plt.imshow(imr)
#plt.show()
plt.figure()
#plt.imshow(imag)
plt.imshow(fibra)
#plt.colorbar()
plt.show()
#%%
#fibra = k #[45:-45,45:-45]
bb = [-15,240,85,530] 

tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=True)
tramos = spl.ordenar_fibra(tramos)
curv,spline = spl.pegar_fibra(tramos,bordes,window=11,s=15)

t_spl = np.linspace(0, 1, 10000)
xf, yf = splev(t_spl, spline)

plt.figure()
plt.imshow(im,cmap='gray')
plt.plot(xf, yf, 'r-')
#plt.colorbar()
plt.show()
plt.figure()
plt.imshow(im,cmap='gray')
#plt.colorbar()
plt.show()
#%%
num = 2000
ni = '{:04d}'.format(num)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0007\ID_0_C1S000700'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float')

imr = im[140:500,700:830] #- gris[140:500,700:830]

plt.figure()
plt.imshow(imr)
plt.show()
#%%
#t1 = time()
a = gabor(imr,0.1)
b = imr - 2*(a[0]-np.min(a[0]) + 10)
c = gaussian(b,0.7)
d = (c - np.mean(c))
e = d * (d<0)
#f = hessian(e)
#g = f * (f<0.7) * (f>0.3)
#h = remove_small_objects((g[:,5:-5])>0,connectivity=1)
#fibra_r = skeletonize(binary_erosion(dilation(h)))

fgi = frangi(-e**3)
fg = binary_closing(fgi==0)
lal = label( remove_small_objects(fg,connectivity=20) )
props = regionprops(lal)
aref = 0
for i in range(np.max(lal)):
    ar = props[i].area
    arb = props[i].extent
    if arb > 0.3: lal[lal==i+1] = 0
#
#fib = skeletonize(lal>0)
#t2 = time()
#print(t2-t1)

#asd = sato(gaussian(e**3,1)) 
#dsa = enhance_contrast(asd/np.max(asd),disk(3))
#esd = remove_small_objects(dsa>80,connectivity=5)
#gfd = esd[:,5:-5]


plt.figure()
plt.imshow( lal )
#plt.colorbar()
plt.show()
#%%
#==============================================================================
# Pruebo forzando tener 1 solo tramo
#==============================================================================
num = 1801
ni = '{:04d}'.format(num)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float')
#
#imr = im[0:60,250:550] #800
imr = im[210:440,570:810] #1800
#imr = im[0:240,380:590] #987
#imr = im[0:280,470:630] #1245
#imr = im[100:400,550:800] #1600
#imr = im[400:700,600:830] #2450
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

# metodo 1
#t1 = time()
#a = gabor(imr,0.1)
#b = imr - 2*(a[0]-np.min(a[0]) + 10)
#c = gaussian(b,0.7)
#d = (c - np.mean(c))
#e = d * (d<0)
#fgi = frangi(-e**3)
#fg = binary_closing(fgi==0)
#lal = label( remove_small_objects(fg,connectivity=20) )
#props = regionprops(lal)
#aref = 0
#for i in range(np.max(lal)):
#    ar = props[i].area
#    arb = props[i].extent
#    if arb > 0.3: lal[lal==i+1] = 0
#lalbc = binary_closing(lal,disk(10))
#fibra1 = skeletonize(lalbc>0)
#t2 = time()
#print('1:',t2-t1)
#
#
##metodo 2
#t1 = time()
#a = gabor(imr,0.1)
#b = imr - 2*(a[0]-np.min(a[0]) + 10)
#c = gaussian(b,0.7)
#d = (c - np.mean(c))
#e = d * (d<0)
#ll = frangi(e)>0.2
#lal = label( remove_small_objects(ll,connectivity=5) )
#props = regionprops(lal)
#aref = 0
#for i in range(np.max(lal)):
#    ar = props[i].area
#    arb = props[i].extent
#    if arb > 0.3: lal[lal==i+1] = 0
#fibra2 = skeletonize(lal>0)
#t2 = time()
#print('2:',t2-t1)

# metodo 3
t1 = time()
a = gabor(imr,0.1)
b = imr - 2*(a[0]-np.min(a[0]) + 10)
c = gaussian(b,0.7)
lp = adjust_log(c,1) - 9
l5 = -lp**5 * (lp<0)
le = enhance_contrast(img_as_ubyte(l5),disk(5)) 

#le = enhance_contrast(img_as_ubyte(lel),disk(3)) 
#mu,st,me = np.mean(le),np.std(le),np.median(le)
#print(mu,st,me,round(mu+st)+1)
##
#
#lal = label(le > round(mu+st))
#props = regionprops(lal)
#
#for i in range(np.max(lal)):
#    ar = props[i].area
#    arb = props[i].extent
##    print(i+1, arb)
#    if arb > 0.31: lal[lal==i+1] = 0                     
##
#lt = binary_closing(lal>0,disk(5)) 
#fibra3 = skeletonize(lt)
#t2 = time()
#print('3:',t2-t1)



#po = hessian(l5)
#op = remove_small_objects(po==1,connectivity=30)
#it = binary_erosion(op)
#at = remove_small_objects(it,min_size=30,connectivity=100)
#ap = dilation(at,disk(3)) 
#lal = label(ap)
#props = regionprops(lal)
#
#for i in range(np.max(lal)):
#    ar = props[i].area
#    arb = props[i].extent
##    print(i+1, arb)
#    if arb > 0.45: lal[lal==i+1] = 0
##ki = binary_closing(lal>0,disk(20))                      
#fibra3 = skeletonize(lal>0)

#yu = sato(-l5)
#uy = enhance_contrast(img_as_ubyte(yu/np.max(yu)),disk(7)) 
#mu,st,me = np.mean(uy),np.std(uy),np.median(uy)
##print(mu,st,me,round(mu+st)+1)
#ty = label(uy>round(mu+st)+1)
#props = regionprops(ty)
#for i in range(np.max(ty)):
#    ar = props[i].area
#    if ar < 10: 
#        ty[ty==i+1] = 0
#          
#lal = label(dilation(ty,disk(3))>0)
#props = regionprops(lal)
#for i in range(np.max(lal)):
#    arb = props[i].extent
#    if arb > 0.65: lal[lal==i+1] = 0
#                      
##    print(i+1, arb)
#fibra3 = skeletonize(binary_closing(lal>0,disk(5)))
#t2 = time()
#print(t2-t1)

lph = -lp*(lp<0)
sas = hessian(lph,mode='nearest')
lal = label(binary_erosion(sas>0.8,disk(1)))  #sas>0.8)
props = regionprops(lal)
for i in range(np.max(lal)):
    arb = props[i].extent
    if arb > 0.3: lal[lal==i+1] = 0
#lap = np.pad(lal,(20,20))
#hg = dilation(lap>0,disk(7))
#hg = binary_closing(lap>0,disk(1))
fibr = skeletonize(dilation(lal>0),method='lee')
#fib = fibr[:,:]>0
#kernel = np.array([[1,1,1],
#                   [1,1,1],
#                   [1,1,1]])
#cf = convolve2d(fib,kernel) # hago la convolución
#convolved_fibra = cf[1:-1,1:-1] * fib
#fic = fib * (convolved_fibra<4)
#lol = label(fic)
#props = regionprops(lol)
#for i in range(np.max(lol)):
#    arb = props[i].area
##    print(i+1, arb)
#    if arb < 7: lol[lol==i+1] = 0
##fibi = ( fib * (lol + (convolved_fibra>3)) )>0
#fibi = ( fib * lol )>0
#fibra = skeletonize(dilation(fibi,disk(3)))
#fre = pcv.morphology.skeletonize(lal>0)
ps, si, so = pcv.morphology.prune(fibr,size=20)
fib = skeletonize(ps>0)
t2 = time()
#print(t2-t1)

lpm = lph[:,:]
lpme, lpst = np.mean(lph),np.std(lph)
#print(lpme,lpst)
lpm[lph < lpme-lpst] = np.mean(lph)
fgg = gaussian(lpm,3)
#err = frangi(-fgg) 
#ert = err/np.max(err)
#er2 = ert**2/np.max(ert**2)
##print(np.mean(er2),np.std(er2))
#ske = skeletonize(er2 > np.mean(er2)/2) #+ np.std(er2))
#lal = label(ske)
#props = regionprops(lal)
#for i in range(np.max(lal)):
#    arb = props[i].extent
##    print(i+1,arb)
##    if arb > 0.1: lal[lal==i+1] = 0
#fib = lal>0


lkj = (fgg -gaussian(fgg,30))
rfm = frangi(-lkj)
rfg = rfm/np.max(rfm)
muu,stt = np.mean(rfg), np.std(rfg)
#print(muu,stt)
ske = skeletonize(rfg > muu+stt*1.3)
lal = label(ske)
props = regionprops(lal)
for i in range(np.max(lal)):
    arb = props[i].extent
    print(i+1,arb)
    if arb > 0.05: lal[lal==i+1] = 0
fib = lal>0

plt.figure()
plt.imshow( fib )
plt.show()
#%%
t1 = time()
bb = [210,570,440,810]  #[210:440,570:810] 1800
#bb = [0,250,60,520] #[0:60,250:520] 800

tramos,bordes = spl.cortar_fibra_rap(fib,bb,cortar_ruido=True)
if len(tramos) > 1 and len(bordes) > 2:
    print('b',len(tramos))
    d0 = 1e6
    for i in range(len(bordes)):
        for j in range(i+1,len(bordes)):
            b1 = np.array(bordes[i])
            b2 = np.array(bordes[j])
            dist = np.sum((b1-b2)**2)
            if dist < d0:
                bb1,bb2 = bordes[i],bordes[j]
                n,m = i,j
                d0 = dist
    conca = tuple([tramos[k] for k in range(len(tramos))])
    tramos = [np.concatenate(conca)]
    bo = np.delete(bordes,[n,m],axis=0)
    nnei = 30
else:
    print('a')
    nnei = 2
    bo = bordes


tramos = ordenar_fibra(tramos,bo,nnei)
curv,spline = spl.pegar_fibra(tramos,bo,window=27,s=25)
t2 = time()
print(t2-t1)

t_spl = np.linspace(0, 1, 10000)
xf, yf = splev(t_spl, spline)

plt.figure()
plt.imshow(b)
plt.plot(xf-bb[1], yf-bb[0], 'r-')
plt.show()
plt.figure()
plt.imshow(b)
plt.plot()
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