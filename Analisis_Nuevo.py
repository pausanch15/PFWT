import numpy as np
import matplotlib.pyplot as plt
import Func_Splines as spl
import Continuacion_Fourier as cf
from skimage.io import imread
from scipy.interpolate import splev
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
from scipy.signal import convolve2d
from skimage.filters import gaussian, frangi, sato, gabor, laplace, sobel, roberts, hessian
from skimage.filters.rank import minimum, enhance_contrast
from skimage.morphology import disk, flood, flood_fill
from skimage.measure import label, regionprops
from skimage.util import img_as_ubyte
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
num = 1800
ni = '{:04d}'.format(num)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float')

imr = im[210:460,570:800] - gris[210:460,570:800]

plt.figure()
plt.imshow(imr)
plt.show()
#%%
t1 = time()
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

fib = skeletonize(lal>0)
t2 = time()
print(t2-t1)

asd = sato(gaussian(e**3,1)) 
dsa = enhance_contrast(asd/np.max(asd),disk(3))
esd = remove_small_objects(dsa>80,connectivity=5)
gfd = esd[:,5:-5]


plt.figure()
plt.imshow( e )
plt.colorbar()
plt.show()
#%%


