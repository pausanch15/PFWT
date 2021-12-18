import numpy as np
import matplotlib.pyplot as plt
import Func_Splines as spl
from skimage.io import imread
from scipy.interpolate import splev
from time import time
#from plantcv import plantcv as pcv
import h5py 
plt.ion()
#%%
from skimage.filters import gaussian, frangi, sato, gabor
from skimage.util import img_as_ubyte
from skimage.filters.rank import minimum, maximum
from skimage.morphology import thin, skeletonize, remove_small_objects, dilation, disk, binary_closing
#%%
def encontrar_fibn(im,bb,nk=5,mima='min',divi=3,size=15, dil=True):
    imr = im[bb[0]:bb[1],bb[2]-20:bb[3]+20]
    a1 = imr - gaussian(imr,20)
    a = gabor(a1,0.1)
    b = a1 - 2*(a[0]-np.min(a[0]) + 10)
    c = b - gaussian(b,30,mode='mirror')
    d = gaussian(c,0.5)
    
    th = gabor(d*(d<0),0.7)
    ttt = th[1]+th[0]
    
    ggg = gaussian(ttt,3,mode='mirror') - gaussian(ttt,4,mode='mirror')
    hi = sato(ggg[:,20:-20],mode='reflect')
    hii = hi / np.max(hi)
    n = nk
    ker = np.zeros((n,n))
    ker[int(n/2),:] = 1
    if mima == 'min':
        mii = minimum( img_as_ubyte(hii),ker)
        gmi = gaussian(mii,3) - gaussian(mii,4)
        fmi = frangi(-gmi)
        fi3 = (fmi/np.max(fmi))**3
        si = np.std(fi3)
    elif mima == 'max':
        mii = maximum( img_as_ubyte(hii),disk(3))
        gmi = gaussian(mii,5) - gaussian(mii,6)
        fmi = frangi(-gmi)
        fi3 = (fmi/np.max(fmi))**3
        si = np.std(fi3)
        
    rmo = remove_small_objects(fi3 > si/divi , min_size=size)
    if dil: drmo = dilation(rmo,disk(10))
    else: drmo = rmo 
    fib = skeletonize( drmo )
    
    lfib = np.where(fib) 
    yma, xma = np.max(lfib[0]),np.max(lfib[1])
    ymi, xmi = np.min(lfib[0]),np.min(lfib[1])
    
    xmax, ymax = min(1023,xma+bb[2]+1), min(1023,yma+bb[0]+1) 
    xmin, ymin = max(0,xmi+bb[2]), max(0,ymi+bb[0])
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

def largo_fib(xf,yf):
    nf = len(xf)
    td = 0
    for i in range(1,nf):
        di = np.sqrt( (xf[i]-xf[i-1])**2 + (yf[i]-yf[i-1])**2  )
        td += di
    return td
#%%
t1 = time()
#    
with h5py.File('fibra.hdf5', 'w') as f:
    h_splf = f.create_group('splines')
    dt = h5py.special_dtype(vlen=np.dtype('float64'))
    
    h_t = h_splf.create_dataset('lista_splf_t',(40,),dtype=dt)
    h_c1 = h_splf.create_dataset('lista_splf_c1',(40,),dtype=dt)
    h_c2 = h_splf.create_dataset('lista_splf_c2',(40,),dtype=dt)
    h_k = h_splf.create_dataset('lista_splf_k',(40,))

    h_bor = f.create_group('bordes')
    h_b1 = h_bor.create_dataset('lista_bordes1',(40,),dtype=dt)
    h_b2 = h_bor.create_dataset('lista_bordes2',(40,),dtype=dt)

    splines, bordes = [], []
    
    bb = [0,100,220,510] #para 760, corre hasta 1100
    #bb = [0,274,411,659] #para 1100, corre hasta 1300 
    #bb= [0,305,484,656] #para 1300, corre hasta 1700 (algunas quedan medio cortas, no mucho)
    # entre 1150 y 1600 aprox, la fibras son muy onduladas (tal vez con un dilation se arregle)
    #bb = [201,474,558,820] #para 1700, corre hasta 2100
    #bb = [332, 552, 525, 835] #para 2100, hasta 2531 (pareciera haber alguna ondulaciones)
    #entre 2531 y 2550 no pude distinguir la fibra de forma rapida
    #bb = [650,850,650,900] #para 2550
    
#    bbt = [bb]
    
    maxi = [1242,1243,1263,1264,1265,1266,1375]
    maxi2 = [1796,1798,1805]
    maxi3 = [1797,1875,2147]
    si0 = [2012,2014,2018,2019,2020,2199]
    
    for num in range(760,800):
        ni = '{:04d}'.format(num)
        ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
        #ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
        im = np.array(ima,dtype='float')
        
        if num%30 == 0: print(num,end=' ')
        
        try:   
            sise,mima,dil = 150, 'min', True
            if num < 1000: dil=False
            if num in maxi: mima, sise = 'max', 30
            if num in range(1300,1700): sise = 80
            if num in range(1700,2100): sise=400
            if num in maxi2: mima, sise = 'max', 350
            if num in maxi3: mima, sise = 'max', 150
            if num in si0: sise = 0
            if num in range(2515,2531): mima, sise = 'min', 900
            if num in range(2531,2550):
#                splines.append(None)
#                bbt.append(None)
                continue
            if num in range(2550,2560): mima, sise = 'min', 1000
            if num > 2560: mima, sise = 'max', 500
            if num == 2567: sise = 310
            
            fib,imr,bbn = encontrar_fibn(im, bb, size=sise, mima=mima, dil=dil)
            tramos,bordes = spl.cortar_fibra_rap(fib,bbn,cortar_ruido=True)
            bo = bordes_reales(tramos,bordes,imprimir=False)
            tramos = spl.ordenar_fibra(tramos)
            curv,spline = spl.pegar_fibra(tramos,bo,window=27,s=25)
    
        except:
            print('Fallo',num)
            break
        
        tf,c1f,c2f,kf = spline[0], spline[1][0], spline[1][1], spline[2]
#        print(tf,c1f,c2f,kf)
        
        if num < 2531: lugf = num-760
        if num > 2531: lugf = num-760-19
        
        h_t[lugf] = tf
        h_c1[lugf] = c1f
        h_c2[lugf] = c2f
        h_k[lugf] = kf
           
        h_b1[lugf] = bo[0]
        h_b2[lugf] = bo[1]
        
#        splines.append(spline)
        
        nb1, nb2, = max(0,bbn[0]-40), min(1024,bbn[2]+40)
        nb3, nb4 = max(0,bbn[1]-40), min(1024,bbn[3]+40)
        bb = [nb1,nb2,nb3,nb4]
        if num == 2530: bb = [650,850,650,900] # para que 2550 no se rompa por el salto
#        bbt.append(bb)
    
t2 = time()
print(t2-t1)
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
n = -1300 + 1600
print(bbt[n])
ni = '{:04d}'.format(1300+n)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float')

    
t_spl = np.linspace(0,1,1000)
xf, yf = splev(t_spl, splines[n])

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
#==============================================================================
# Probando 1 sola fibra 
#==============================================================================
num = 800

ni = '{:04d}'.format(num)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float') 

bbns = {800:[0,60,250,550], 819:[0,70,220,550], 1242:[0,280,450,650], 824:[0,122,224,524],
828:[0,140,234,532], 760:[0,100,220,510], 1800:[210,440,570,810], 1243:[0,280,450,650],
1308:[0, 300, 450, 650], 911:[0,197,326,627], 965:[0,257,364,597], 1300:[0,305,484,656]}

bb = bbns[num]
#bb = bbt[num-2550]
#bb = bbt[-1]
#bb = [650,850,650,900]

imr = im[bb[0]:bb[1],bb[2]-20:bb[3]+20]

plt.figure()
plt.imshow(imr)
plt.show()
#%%
a1 = imr - gaussian(imr,20)
a = gabor(a1,0.1)
b = a1 - 2*(a[0]-np.min(a[0]) + 10)
c = b - gaussian(b,30,mode='mirror')
d = gaussian(c,0.5)

th = gabor(d*(d<0),0.7)
ttt = th[1]+th[0]

ggg = gaussian(ttt,3,mode='mirror') - gaussian(ttt,4,mode='mirror')
hi = sato(ggg[:,20:-20],mode='reflect')
hii = hi / np.max(hi)
n = 5
ker = np.zeros((n,n))
ker[int(n/2),:] = 1
mii = minimum( img_as_ubyte(hii),ker)
maa = maximum( img_as_ubyte(hii),disk(3))
fgi = gaussian(mii,3)

gmi = gaussian(mii,3) - gaussian(mii,4)
gma = gaussian(maa,5) - gaussian(maa,6)

fmi = frangi(-gmi)
fma = frangi(-gma)

fi3 = (fmi/np.max(fmi))**3
fa3 = (fma/np.max(fma))**3 
ma,sa = np.mean(fa3), np.std(fa3)
mi,si = np.mean(fi3), np.std(fi3)
print(mi,si)
rmo = remove_small_objects(fi3 > si/3 , min_size=180)
rma = remove_small_objects(fa3 > sa/3 , min_size=310)

nd = 1
ytr = dilation(rmo,disk(nd))

plt.figure()
plt.imshow( skeletonize(ytr) )
#plt.colorbar()
plt.show()
#plt.figure()
#plt.imshow( dilation(rma,disk(3)) )
##plt.colorbar()
#plt.show()
#plt.figure()
#plt.imshow( mea )
##plt.colorbar()
#plt.show()

#%%
#==============================================================================
# Traigo hdf5
#==============================================================================
splif = []
bor = []
with h5py.File('fibra.hdf5', 'r') as f:  
    gspf = f.get('splines')
    tf_h, kf_h = gspf['lista_splf_t'], gspf['lista_splf_k']
    c1f_h, c2f_h = gspf['lista_splf_c1'], gspf['lista_splf_c2'] 
    
    gbor = f.get('bordes')
    b1_h,b2_h = gbor['lista_bordes1'], gbor['lista_bordes2']
        
    for i in range(len(tf_h)):
#        imag.append(im_h[i])
        splif.append([tf_h[i],[c1f_h[i],c2f_h[i]],kf_h[i]])
        bor.append(np.array([b1_h[i],b2_h[i]]))
#%%
n = 769 - 760
ni = '{:04d}'.format(760+n)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float')

    
t_spl = np.linspace(0,1,1000)
xf, yf = splev(t_spl, splif[n])
xb,yb = bor[n][:,0], bor[n][:,1] 

plt.figure()
plt.imshow(im)
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(im)
plt.plot(xf, yf, 'r-')
plt.plot(xb, yb, 'ko')
plt.show()