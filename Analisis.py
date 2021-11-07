import numpy as np
import matplotlib.pyplot as plt
import Func_Splines as spl
import Continuacion_Fourier as cf
from skimage.io import imread
from scipy.interpolate import splev
import h5py 
#%%
# traigo imagenes
gris = np.zeros((1024,1024))
for i in range(1,301):
    ni = '{:04d}'.format(i)
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\Grises\ID_0_C1S000600'+ni+'.tif')
    ima = np.array(ima,dtype='float')
    gris += ima
gris = gris/300

ref = np.zeros((1024,1024))
for i in range(1,301):
    ni = '{:04d}'.format(i)
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\Referencia\ID_0_C1S000300'+ni+'.tif')
    ima = np.array(ima,dtype='float')
    ref += ima
ref = ref/300
#%%
# para ver
plt.figure()
plt.imshow(ref-gris)
plt.show()
#%%
# reconstruyo fibras y hago FTP
from time import time
dphs, splines = [],[]
t1 = time()
#47 no funca, 51 usar mult=2 
#probs 1-500: 47, 51
for num in range(1,501): #
    if num%20 == 0: print(num,end=' ')
    ni = '{:04d}'.format(num)
    im = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\FTP_fibra\ID_0_C1S000500'+ni+'.tif')
    im = np.array(im,dtype='float')
    
    mul = 2.46
    if num == 51: mul = 2
    elif num == 47: continue
    fibra, bb, fib_an = spl.recupera_fibra(im,gris,std_mul=mul)
    tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=True)
    tramos = spl.ordenar_fibra(tramos)
    try:
        curv,spline = spl.pegar_fibra(tramos,bordes,window=15,s=16)
        splines.append(spline)
        imfc, ffc = cf.hacer_fou_cont(im,fibra,bb,ventana=150,mmx=2,mmy=2)
        thx,thy, ns = 0.5, 80, 0.5
        dph, ft, gf = cf.dphase_2d(imfc-gris,ref-gris,thx,thy,ns,inde=9)
        dphs.append(dph)
    except: print('Jajaja', num)
t2 = time()
t2-t1
#%%
# ms visualizacion
i = 9
dpd = dphs[i]
t_spl = np.linspace(0, 1, 10000)
xf, yf = splev(t_spl, splines[i])
lim = 10
plt.figure()
plt.imshow(dpd[lim:-lim,lim:-lim])
plt.colorbar()
#plt.show()
#plt.figure()
#plt.imshow(imfc-gris,cmap='gray')
#plt.show()
#plt.figure()
#plt.imshow(im-gris,cmap='gray')
plt.plot(xf-lim, yf-lim, 'r-')
#plt.colorbar()
plt.show()
#%%
# guardo en hdf5
tf,c1f,c2f,kf = [],[],[],[]
for i in range(len(splines)):
    tf.append(splines[i][0])
    c1f.append(splines[i][1][0])
    c2f.append(splines[i][1][1])
    kf.append(splines[i][2])
    
with h5py.File('splines.hdf5', 'w') as f:
    h_splf = f.create_group('splines_recons')
    dt = h5py.special_dtype(vlen=np.dtype('float64'))
    
    h_splf.create_dataset('lista_splf_t',data=tf,dtype=dt)
    h_splf.create_dataset('lista_splf_c1',data=c1f,dtype=dt)
    h_splf.create_dataset('lista_splf_c2',data=c2f,dtype=dt)
    h_splf.create_dataset('lista_splf_k',data=kf)

with h5py.File('ftp.hdf5', 'w') as f:
    h_im = f.create_group('dif_phase')
    h_im.create_dataset('dp',data=dphs)
    
#%%
# traigo hdf5
dps, splif = [],[]
spli_hdf = r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\Analisado\splines.hdf5'
ftp_hdf = r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\Analisado\ftp.hdf5'

with h5py.File(spli_hdf, 'r') as f:  
    gspf = f.get('splines_recons')
    tf_h, kf_h = gspf['lista_splf_t'], gspf['lista_splf_k']
    c1f_h, c2f_h = gspf['lista_splf_c1'], gspf['lista_splf_c2'] 
        
    for i in range(len(tf_h)):
#        imag.append(im_h[i])
        splif.append([tf_h[i],[c1f_h[i],c2f_h[i]],kf_h[i]])
with h5py.File(ftp_hdf, 'r') as f:
    gdp = f.get('dif_phase')
    h_dp = gdp['dp']
    for i in range(len(h_dp)):
        dps.append(h_dp[i])
#%%
i = 0
dpd = dps[i]
t_spl = np.linspace(0, 1, 10000)
xf, yf = splev(t_spl, splif[i])
lim = 10

plt.figure()
plt.imshow(dpd[lim:-lim,lim:-lim])
plt.plot(xf-lim, yf-lim, 'r-')
plt.colorbar()
plt.show()
#%%
# Para hacer una imagen sola
num = 324 #47,51
# 47 no funciona
ni = '{:04d}'.format(num)
im = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\FTP_fibra\ID_0_C1S000500'+ni+'.tif')
im = np.array(im,dtype='float')

mul = 2.46
fibra, bb, fib_an = spl.recupera_fibra(im,gris,std_mul=mul)
tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=True)
tramos = spl.ordenar_fibra(tramos)
try:
    curv,spline = spl.pegar_fibra(tramos,bordes,window=15,s=16)
except: print('Jajaja', num)

imfc, ffc = cf.hacer_fou_cont(im,fibra,bb,ventana=150,mmx=2,mmy=2)
thx,thy, ns = 0.5, 80, 0.5
dph, ft, gf = cf.dphase_2d(imfc-gris,ref-gris,thx,thy,ns,inde=9)
#%%
t_spl = np.linspace(0, 1, 10000)
xf, yf = splev(t_spl, spline)
ventana = 150
bb1,bb2 = int((bb[0]+bb[2])/2), int((bb[1]+bb[3])/2)
vent = int(ventana/2)
b1,b2,b3,b4 = bb1-vent,bb1+vent,bb2-vent,bb2+vent
#im_nf = im[bb[0]:bb[2], bb[1]:bb[3]] * (1-fib_an)
#im_nf[im_nf==0] = np.nan
#def_hole = np.zeros(np.shape(im))
#def_hole += im
#def_hole[bb[0]:bb[2], bb[1]:bb[3]] = im_nf
#def_hole = def_hole[b1:b2,b3:b4] 

#%%
from skimage.filters import laplace, sobel
plt.figure()
imlap = sobel(laplace(im-0.8*gris,ksize=3),mode='wrap')
binariza = np.mean(imlap) + 2.46*np.std(imlap)
imb = imlap>binariza
plt.imshow(imlap[b1:b2,b3:b4],cmap='gray')
#plt.imshow(laplace(im-0.8*gris,ksize=3),cmap='gray')
#plt.imshow(imb,cmap='gray')
#plt.plot(xf,yf,'r-')
plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
#plt.colorbar()
plt.savefig('imageni.png',bbox_inches='tight')  
plt.show()
#%%
t_spl = np.linspace(0, 1, 10000)
xf, yf = splev(t_spl, spline)
plt.figure()
plt.imshow(im,cmap='gray')
plt.plot(xf,yf,'r-')
plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
plt.savefig('imageni.png',bbox_inches='tight')  
plt.show()
#%%
t_spl = np.linspace(0, 1, 10000)
xf, yf = splev(t_spl, spline)
plt.figure()
plt.imshow(im-0.8*gris)
plt.plot(xf, yf, 'r-')
plt.colorbar()
plt.show()