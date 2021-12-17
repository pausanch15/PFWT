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
from scipy.signal import convolve2d, medfilt
from skimage.filters import gaussian, frangi, sato, gabor, laplace, sobel, roberts, hessian, inverse
from skimage.filters.rank import minimum, enhance_contrast, gradient
from skimage.morphology import disk, flood, flood_fill
from skimage.measure import label, regionprops
from skimage.util import img_as_ubyte
from skimage.exposure import adjust_gamma, adjust_log
from skimage.morphology import thin, skeletonize, remove_small_objects, dilation, binary_erosion, binary_closing, closing
#%%
num = 1500
ni = '{:04d}'.format(num)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0007\ID_0_C1S000700'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float')

imr = im[0:370,850:950]

plt.figure()
plt.imshow(im)
plt.show()
plt.figure()
plt.imshow(imr)
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
#[50,450,650,850]

bb = [0,370,850,950]
imr = im[bb[0]:bb[1],bb[2]:bb[3]]

prun=20
std_mul=2
disco=7
mulm=0.8
tama=145
tipog='wrap'

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
   
plt.figure()
plt.imshow(b - gaussian(b,30,mode='mirror'))
plt.show()
#%%
num = 819
ni = '{:04d}'.format(num)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float')

bbns = {800:[0,60,250,550], 819:[0,70,220,550], 1242:[0,280,450,650], 824:[0,122,224,524],
828:[0,140,234,532], 760:[0,100,220,510], 1800:[210,440,570,810], 1243:[0,280,450,650] }
bb = bbns[num]
imr = im[bb[0]:bb[1],bb[2]:bb[3]]

plt.figure()
plt.imshow(im)
plt.show()
plt.figure()
plt.imshow(imr)
plt.show()
#%%

a = gabor(imr,0.1)
b = imr - 2*(a[0]-np.min(a[0]) + 10)
c = b - gaussian(b,30,mode='mirror')
d = gaussian(c,0.5)

th = gabor(d*(d<0),0.7)
ttt = th[1]+th[0]
#rrr = frangi(ttt)
#
#trt = gaussian(rrr,3,mode='constant') - gaussian(rrr,50,mode='constant')
#fff = enhance_contrast(img_as_ubyte(trt),disk(7))
#fff[:,:10] = 0
#fff[:,-10:] = 0
#thee = 2
#kse = fff * (fff>thee)
#muu, stt, mme = np.mean(kse[kse>0]), np.std(kse[kse>0]), np.median(kse[kse>0])
#print(muu, stt, mme)
#thr = muu + stt * 0.3
##thr = mme - stt
#lal = label(kse>0)
#props = regionprops(lal)
#for i in range(np.max(lal)):
#    ori = props[i].orientation
##    arb = props[i].area
#    mma = np.max(kse[lal==i+1])
#    print(i+1,'\t',mma) #,'\t',arb)
#    if mma < thr or np.abs(ori) < 0.04: lal[lal==i+1] = 0
#fib = skeletonize(lal>0)

#sd = gaussian(ttt,5,mode='wrap')
#fro = sd - gaussian(ttt,20,mode='wrap')
#lpf = laplace(fro)
#dor = lpf+np.min(lpf) / np.max(lpf+np.min(lpf))
#der = frangi(dor)
#rer = enhance_contrast(img_as_ubyte(der/np.max(der)),disk(10))
#moo, soo = np.mean(rer[rer>0]), np.mean(rer[rer>0])
#thr = moo+soo
#print(thr)

#fred = lpf**2 * np.sign(lpf)
#redf  = frangi(fred)[:,5:-5]
#
#greg = fro**2 * np.sign(fro)
#reti = sato(greg)[:,10:-10]
#ehan = enhance_contrast(reti/np.max(reti),disk(10))
#moo, soo = np.mean(ehan), np.mean(ehan)
#thr = moo+soo
#print(thr)
#qwer = laplace(reti)
#derf = sato(-qwer)
#rtyu = enhance_contrast(derf/np.max(derf),disk(5))

#hju = fro**2 * np.sign(fro)
#popp = enhance_contrast(-hju/np.max(-hju),disk(5))

#cvb = gaussian(ttt,0.5)
#vbn = (cvb - np.mean(cvb)) / np.max(np.abs(cvb))
df = d * (d<0)
df[:,:10], df[:,-10:] = 0,0
#vbn = (df - np.mean(df)) / np.max(np.abs(df)+0.1)
#dfgt = gaussian(-vbn,0.2,mode='mirror') - gaussian(-vbn,5,mode='mirror')
#mjk = enhance_contrast(-vbn,disk(10))

#y = gaussian(dfgt,5,mode='wrap') 
#zo = (y - gaussian(dfgt,15))**3 

     
dert = gaussian(frangi(df),1) - gaussian(frangi(df),2)
erw = gaussian(dert,3)**2
mjk = frangi(-gaussian(erw,5))

#plt.figure()
#plt.imshow( reti )
#plt.colorbar()
#plt.show()
plt.figure()
plt.imshow( frangi(df) )
plt.colorbar()
plt.show()
plt.figure()
plt.imshow( (mjk/np.max(mjk))   )
#plt.colorbar()
plt.show()
#plt.figure()
#plt.imshow( df )
#plt.show()
#plt.figure()
#plt.imshow( rer )
#plt.show()


#%%
def encontrar_fib(im,bb,muls=0.5,thee=2,orie=0.04):
    imr = im[bb[0]:bb[1],bb[2]:bb[3]]
    a = gabor(imr,0.1)
    b = imr - 2*(a[0]-np.min(a[0]) + 10)
    c = b - gaussian(b,30,mode='mirror')
    d = gaussian(c,0.5)
    
    th = gabor(d*(d<0),0.75)
    ttt = th[1]+th[0]
    rrr = frangi(ttt)
    
    trt = gaussian(rrr,3,mode='constant') - gaussian(rrr,50,mode='constant')
    fff = enhance_contrast(img_as_ubyte(trt),disk(7))
    fff[:,:5] = 0
    fff[:,-5:] = 0
    kse = fff * (fff>thee)
    muu, stt, mme = np.mean(kse[kse>0]), np.std(kse[kse>0]), np.median(kse[kse>0])
#    print(muu, stt, mme)
    thr = muu + stt * muls
    #thr = mme - stt
    lal = label(kse>0)
    props = regionprops(lal)
    for i in range(np.max(lal)):
        ori = props[i].orientation
        mma = np.max(kse[lal==i+1])
#        print(i+1,'\t',mma) #,'\t',arb)
        if mma < thr or np.abs(ori) < orie: lal[lal==i+1] = 0
    fib = skeletonize(lal>0)
    
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

def bordes_reales(tramos,bordes,imprimir=False):
    if len(tramos) > 1 and len(bordes) > 2:
        if imprimir: print('n° tramos:',len(tramos),'\nn° bordes:',len(bordes))
        d0 = 0
        for i in range(len(bordes)):
            for j in range(i+1,len(bordes)):
                b1 = np.array(bordes[i])
                b2 = np.array(bordes[j])
                dist = np.sum((b1-b2)**2)
                if dist > d0:
                    n,m = i,j
                    d0 = dist
        bo = np.array([bordes[n],bordes[m]])
        return bo
    else:
        if imprimir: print('n° tramos:',len(tramos),'\nn° bordes:',len(bordes))
        bo = bordes
        return bo
#%%
num = 824
ni = '{:04d}'.format(num)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float')

bbns = {800:[0,60,250,550], 819:[0,70,220,550], 1243:[0,280,450,650], 824:[0,122,224,524]}
bb = bbns[num]

t1 = time()
fib,imr,bbn = encontrar_fib(im,bb,muls=0.7)
tramos,bordes = spl.cortar_fibra_rap(fib,bbn,cortar_ruido=True)
bo = bordes_reales(tramos,bordes,imprimir=False)
tramos = spl.ordenar_fibra(tramos)
curv,spline = spl.pegar_fibra(tramos,bo,window=27,s=25)
t2 = time()
print(t2-t1)

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
t1 = time()
splines = []
bb = [0,100,220,510] #para 760
bbt = [bb]
for num in range(760,1100):
    ni = '{:04d}'.format(num)
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
    #ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
    im = np.array(ima,dtype='float')
    
    if num%30 == 0: print(num,end=' ')
    
    try:
        muls = 0.3
        if num == 819: muls=-0.1
        
        fib,imr,bbn = encontrar_fib(im,bb,muls=muls,thee=2,orie=0.04)
        tramos,bordes = spl.cortar_fibra_rap(fib,bbn,cortar_ruido=True)
        bo = bordes_reales(tramos,bordes,imprimir=False)
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
n = -760 + 828
print(bbt[n])
ni = '{:04d}'.format(760+n)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float')

t_spl = np.linspace(0, 1, 10000)
xf, yf = splev(t_spl, splines[n])

plt.figure()
plt.imshow(im)
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(im)
plt.plot(xf, yf, 'r-')
plt.show()
#%%
