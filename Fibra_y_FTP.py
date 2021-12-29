import numpy as np
import matplotlib.pyplot as plt
import Func_Splines as spl
import Continuacion_Fourier as cf
from skimage.io import imread
from scipy.interpolate import splev
from time import time
import h5py
from tqdm import tqdm 
#from plantcv import plantcv as pcv
plt.ion()
#%%
#==============================================================================
# Reconstrucción de la fibra.
#==============================================================================
t1 = time()
#    
with h5py.File('fibra.hdf5', 'w') as f:
    h_splf = f.create_group('splines')
    dt = h5py.special_dtype(vlen=np.dtype('float64'))
    
    h_t = h_splf.create_dataset('lista_splf_t',(1921,),dtype=dt)
    h_c1 = h_splf.create_dataset('lista_splf_c1',(1921,),dtype=dt)
    h_c2 = h_splf.create_dataset('lista_splf_c2',(1921,),dtype=dt)
    h_k = h_splf.create_dataset('lista_splf_k',(1921,))

    h_bor = f.create_group('bordes')
    h_b1 = h_bor.create_dataset('lista_bordes1',(1921,),dtype=dt)
    h_b2 = h_bor.create_dataset('lista_bordes2',(1921,),dtype=dt)

    splines, bordes = [], []
    
    bb = [0,100,220,510] #para 760, corre hasta 1100
    #bb = [0,274,411,659] #para 1100, corre hasta 1300 
    #bb= [0,305,484,656] #para 1300, corre hasta 1700 (algunas quedan medio cortas, no mucho)
    # entre 1150 y 1600 aprox, la fibras son muy onduladas (tal vez con un dilation se arregle)
    #bb = [201,474,558,820] #para 1700, corre hasta 2100
    #bb = [332, 552, 525, 835] #para 2100, hasta 2531 (pareciera haber alguna ondulaciones)
    #entre 2531 y 2550 no pude distinguir la fibra de forma rapida
    #bb = [650,850,650,900] #para 2550, hasta 2700
    
#    bbt = [bb]
    
    maxi = [1242,1243,1263,1264,1265,1266,1375]
    maxi2 = [1796,1798,1805]
    maxi3 = [1797,1875,2147]
    si0 = [2012,2014,2018,2019,2020,2199]
    
    for num in range(760,2700):
        ni = '{:04d}'.format(num)
        ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
        #ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
        im = np.array(ima,dtype='float')
        
        if num%50 == 0: print(num,end=' ')
        
        try:   
            sise,mima,dil = 150, 'min', True
            if num < 1000: dil=False
            if num in maxi: mima, sise = 'max', 30
            if num == 1264: sise = 80
            if num in range(1300,1700): sise = 80
            if num in range(1700,2100): sise = 400
            if num in maxi2: mima, sise = 'max', 350
            if num in maxi3: mima, sise = 'max', 150
            if num in si0: mima,sise = 'min', 10
            if num in range(2515,2531): mima, sise = 'min', 900
            if num in range(2531,2550):
#                splines.append(None)
#                bbt.append(None)
                continue
            if num in range(2550,2560): mima, sise = 'min', 1000
            if num > 2560: mima, sise = 'max', 500
            if num == 2567: sise = 310
            
            fib,imr,bbn = spl.encontrar_fibn(im, bb, size=sise, mima=mima, dil=dil)
            tramos,bordes = spl.cortar_fibra_rap(fib,bbn,cortar_ruido=True)
            bo = spl.bordes_reales(tramos,bordes,imprimir=False)
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
#==============================================================================
# Perfil de alturas
#==============================================================================
#Traigo imagenes
gris = np.zeros((1024, 1024))
for i in range(1, 16):
    ni = '{:04d}'.format(i)
#    ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0003\ID_0_C1S000300'+ni+'.tif')
    ima = np.array(ima, dtype='float')
    gris += ima
gris = gris/15

ref = np.zeros((1024, 1024))
for i in range(1, 14):
    ni = '{:04d}'.format(i)
#    ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0002/ID_0_C1S000300'+ni+'.tif')
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0002\ID_0_C1S000300'+ni+'.tif')
    ima = np.array(ima, dtype='float')
    ref += ima
ref = ref/13
#%%
#Saco altura para las imagenes que tienen la fibra
#Hago ftp y guardo en hf5 las alturas
thx,thy, ns = 0.25, 45, 0.75 #0.5, 80, 0.5

#Defino los valores de L y D (los medimos en el labo)
d = (1014-9) / 81 * 0.02086
w = 2*np.pi/d
L, D = 79.6, 20.3

#with h5py.File('18-12-2021_ftp.hdf5', 'w') as f:
with h5py.File('alturas.hdf5', 'w') as f: #le cambio el nombre para mi    
    h_im = f.create_group('hs')
    alt = h_im.create_dataset('alts', shape=(3072,1024,1024))
    for num, i in zip(range(1, 3073), tqdm(range(1, 3073))):
        ni = '{:04d}'.format(num)
#        ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0001/ID_0_C1S000100'+ni+'.tif')
        ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
        im = np.array(ima,dtype='float')
        dph, ft, gf = cf.dphase_2d(im,ref-gris,thx,thy,ns,inde=9)
        altura = (L*dph) / (dph - w*D)
        alt[num-1] = altura - np.mean(altura)
        del(ni, ima, im, altura)
#%%
