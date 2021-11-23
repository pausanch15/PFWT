import numpy as np
import matplotlib.pyplot as plt
import Func_Splines as spl
import Continuacion_Fourier as cf
from skimage.io import imread
#from scipy.interpolate import splev
import h5py 
#%%
dps = []
ftp_hdf = r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\Analisado\ftp.hdf5'
#hay un ftp2 tambien
with h5py.File(ftp_hdf, 'r') as f:
    gdp = f.get('dif_phase')
    h_dp = gdp['dp']
    for i in range(len(h_dp)):
        dps.append(h_dp[i])
#%%
plt.figure()
plt.imshow(dps[0][300:900,300:900])
plt.show()
plt.figure()
plt.imshow(dps[0])
plt.show()
#%%
nt, nxy = 300, 500
dps_c = np.zeros((nt,nxy,nxy))
l1, l2 = 300, 800
for i in range(nt):
    dps_c[i] = dps[i][l1:l2,l1:l2] - np.mean(dps[i][l1:l2,l1:l2])
#%%
dpft = np.fft.fftn(dps_c)
w, kx, ky = np.fft.fftfreq(nt,1/250),0.159* np.fft.fftfreq(nxy,0.2106), 0.159*np.fft.fftfreq(nxy,0.2106) 
#%%
l = 300*0.2106
lam_max = 2*l
k_m = 2*np.pi/lam_max
k = np.linspace(0,100*k_m,150)
#%%
w2 = 2*np.pi*kx[:150] * 9810 * np.tanh(2*np.pi*kx[:150] * 13)*(1+ (2*np.pi*kx[:150])**2 * (1/0.369)**2)
#w2 = k * 9810 * np.tanh(k * 13) 
ws = np.sqrt(w2)

#plt.figure()
#plt.plot(kx[:150],ws)
#plt.show()
#%%
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

kxx, kyy = np.meshgrid(kx,ky)
r,theta = cart2pol(kxx,kyy)
#%%
n = 56
#ext = 2*np.pi*np.array([np.min(kx),np.max(kx),np.min(ky),np.max(ky)])
ext = 2*np.pi * np.array([np.min(kx),np.max(kx),np.min(w),np.max(w)])
fig = plt.figure()
plt.imshow(np.log(np.abs(np.fft.fftshift(dpft[:,:,n])))**2, extent=ext,aspect='auto')
#plt.plot(kx[:150]*2*np.pi,ws,'r-')
#plt.plot(-kx[:150],-ws,'r--')
#plt.plot(-kx[:150],ws,'r--')
#plt.plot(kx[:150],-ws,'r--')
#plt.plot(kx[:150],ws,'r--')
plt.colorbar()
plt.xlabel('kx')
plt.ylabel('w')
plt.ylim([-780,780])
plt.show()
#%%
n = 29
ext = 2*np.pi*np.array([np.min(kx),np.max(kx),np.min(ky),np.max(ky)])
fig = plt.figure()
plt.imshow(np.log(np.abs(np.fft.fftshift(dpft[n,:,:])))**2, extent=ext,aspect='auto',vmax=100)
plt.colorbar()
plt.xlabel('kx')
plt.ylabel('w')
#plt.ylim([-780,780])
plt.show()

#%%
# lo de promediar en radio 
rs = np.fft.fftshift(r)
r1 = np.logical_and(rs >= 0.354, rs < 0.357)
r2 = np.logical_and(rs >= 0.357, rs < 0.360)
a = 1*r1 + 2*r2
plt.figure()
ext = np.array([np.min(kx),np.max(kx),np.min(ky),np.max(ky)])
plt.imshow(a,extent=ext)
plt.colorbar()
plt.show()
#%%
kw = np.zeros((120,300))

rs = np.fft.fftshift(r)
ris = np.arange(0,0.363,0.003)
rim = (ris[1:]+ris[:-1])/2 
      
dpsh = np.fft.fftshift(dpft)
for n in range(300):
#n = 0 
#    dpfts = np.abs(np.fft.fftshift(dpft[n,:,:]))
    dpfts = np.abs(dpsh[n,:,:])
    
    vmr = []
    for i in range(1,len(ris)):
        r1 = np.logical_and(rs > ris[i-1], rs < ris[i])
        mr_dp = np.mean( (np.log(dpfts)[r1>0])**2 )
        vmr.append(mr_dp) 
        
    kw[:,n] = vmr
    
plt.figure()
plt.plot(rim,vmr)
plt.show()      
#%%
plt.figure()
ext = 2*np.pi*np.array([np.min(w)/1000,np.max(w)/1000,np.min(rim),np.max(rim)])
plt.imshow(kw, origin='lower', extent=ext)
plt.show()
#%%
ext = 2*np.pi*np.array([np.min(kx),np.max(kx),np.min(ky),np.max(ky)])
fig = plt.figure()
plt.imshow(np.log(dpfts)**2*r1, extent=ext,aspect='auto') #,vmax=100)
plt.colorbar()
plt.xlabel('kx')
plt.ylabel('w')
#plt.ylim([-780,780])
plt.show()

#%%
#-------------------------------------------------------------------------------------------
#%%
# Esto lo uso para probar distintas convinaciones de thx, thy, ns para reducir las lineas verticales
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
num = 10
ni = '{:04d}'.format(num)
im = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\FTP_fibra\ID_0_C1S000500'+ni+'.tif')
im = np.array(im,dtype='float')
mul = 2.46
if num == 51: mul = 2
#elif num == 47: continue
fibra, bb, fib_an = spl.recupera_fibra(im,gris,std_mul=mul)
imfc, ffc = cf.hacer_fou_cont(im,fibra,bb,ventana=150,mmx=2,mmy=2)
thx,thy, ns = 0.25, 45, 0.75
dph, ft, gf = cf.dphase_2d(imfc-gris,ref-gris,thx,thy,ns,inde=9)
#%%
lim = 15
plt.figure()
plt.imshow(dph[lim:-lim,lim:-lim])
#plt.imshow(dph[300:900,300:900])
plt.title('thx = '+str(thx)+', thy = '+str(thy)+', ns = '+str(ns))
#plt.imshow(dph)
plt.colorbar()
plt.show()
#%%
#----------------------------------------------------------------------------------------------------
#Pruebo de hacer con todos los datos
dps = []
ftp_hdf = r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\Analisado\ftp.hdf5'
#hay un ftp2 tambien
with h5py.File(ftp_hdf, 'r') as f:
    gdp = f.get('dif_phase')
    h_dp = gdp['dp']
    for i in range(len(h_dp)):
        dps.append(h_dp[i])
#%%
dp_fft = np.zeros((1004,1004,100)) + 1j * np.zeros((1004,1004,100))
for i in range(100):
    if i%10 == 0: print(i, end=' ')
    dp_fft[:,:,i] = np.fft.fft2(dps[i][10:-10,10:-10])
#%%
#499 = len(dps)
#1004 para sacar 10 pixeles de cada borde
from time import time
t1 = time()
ft_espacial = np.zeros((1004,1004)) + 1j * np.zeros((1004,1004))
with h5py.File('ener_hdf.hdf5', 'w') as f:
    esp_ft = f.create_group('espacial')
    dats = esp_ft.create_dataset('llenar',(1004,1004,499), dtype='complex')
    for j in range(4):
        dp_fft = np.zeros((1004,1004,100)) + 1j * np.zeros((1004,1004,100))
        for i in range(100):
    #        if i%10 == 0: print(i, end=' ')
            dp_fft[:,:,i] = np.fft.fft2(dps[100*j+i][10:-10,10:-10])
        dats[:,:,j*100:100*j+100] = dp_fft
     
    dp_fft = np.zeros((1004,1004,99)) + 1j * np.zeros((1004,1004,99))
    for i in range(99):
#        if i%10 == 0: print(i, end=' ')
        dp_fft[:,:,i] = np.fft.fft2(dps[400+i][10:-10,10:-10])
    dats[:,:,400:499] = dp_fft
    
        
#    for i in range(499):
#        if i%20 == 0: print(i,end=' ')
#        ftes = np.fft.fft2(dps[i][10:-10,10:-10]) 
#        dats[:,:,i] = np.fft.fftshift(ftes)

t2 = time()
print(t2-t1)
# me llevo 3120 seg correr, estaba haciendo otras cosas tambien asi que puede ser que tarde un poco menos
#%%
#todavia estoy viendo como hacer funcionar esto
t1 = time()
with h5py.File('enert_hdf.hdf5', 'w') as fw:
    tem_ft = fw.create_group('ft_total')
    tf = tem_ft.create_dataset('transformada',(1004,1004,499), dtype='complex')
    with h5py.File('ener_hdf','r') as fr:
        ftesp = fr.get('espacial')
        dfft = ftesp['llenar'] 
        for ii in range(1004):
            if ii%20 == 0: print(ii, end=' ')
            for jj in range(1004):
                # print(jj)
#                st = ftesp['dp'][ii, jj, :]
                ftij = np.fft.fft(dfft[ii, jj, :])   
                # ftij = np.fft.fft(st[ii,jj,:])   
                tf[ii, jj, :] = ftij
#        ftij = np.fft.fft(dfft[:,0,2]) 
#        for i in range(len(ftij)):
#            tf[0,2,i] = ftij[i] 
#        tf[0,0,:] = np.fft.fft(dfft[:,0,0]) 
#        print( np.fft.fft(dfft[:,0,0]), len( np.fft.fft(dfft[:,0,0] ) ) )
#        for i in range(400):
#            if (i+0)%20 == 0: print(i, end=' ')
#            for j in range(400):
###                print(j)
#                ftij = np.fft.fft(dfft[:,i,j])   
#                for k in range(len(ftij)):
#                    tf[i,j,k] = ftij[k]


t2 = time()
print(t2-t1)    
# Esta parte tardo 624 seg (sin estar haciendo nada mas con la compu)
#ft_esp_tem = np.zeros((499,1004,1004))
#%%
w, kx, ky = 2*np.pi*np.fft.fftfreq(499,1/250), 2*np.pi*np.fft.fftfreq(1004,0.2106), 2*np.pi*np.fft.fftfreq(1004,0.2106)

w2 = kx * 9810 * np.tanh(kx * 13) * (1 + (kx)**2 * (1/0.369)**2)
ws = np.sqrt(w2)

with h5py.File('enert_hdf', 'r') as f:
    grup = f.get('ft_total')
    datos = grup['transformada']
#    n = 18
#    wk = datos[:,n,:]

#    ext = np.array([np.min(kx),np.max(kx),np.min(w),np.max(w)])
#
#    plt.figure()
##    plt.imshow(wk)
#    plt.imshow((np.log(np.abs(np.fft.fftshift(wk)))**2).T,aspect='auto',origin='lower',extent=ext)
##    plt.plot(-kx,-ws,'r--')
##    plt.plot(-kx,ws,'r--')
##    plt.plot(kx,-ws,'r--')
##    plt.plot(kx,ws,'r--')
#    plt.xlim(-15,15)
#    plt.ylim(-780,780)
#    plt.show()

    t1 = time()
    for nw in [59,60,61,62,63]:
#        print('aca')
        kxky = datos[:,:,nw]
        a = np.mean(kxky)
#        print('lo pase')
    t2 = time()
    print(t2-t1)
    ext = np.array([np.min(kx),np.max(kx),np.min(kx),np.max(kx)])

    plt.figure()
#    plt.imshow(wk)
    plt.imshow((np.log(np.abs(np.fft.fftshift(kxky)))**2).T,aspect='auto',origin='lower',extent=ext)
#    plt.plot(-kx,-ws,'r--')
#    plt.plot(-kx,ws,'r--')
#    plt.plot(kx,-ws,'r--')
#    plt.plot(kx,ws,'r--')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.show()
#%%
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

w, kx, ky = 2*np.pi*np.fft.fftfreq(499,1/250), 2*np.pi*np.fft.fftfreq(1004,0.2106), 2*np.pi*np.fft.fftfreq(1004,0.2106)
kxx, kyy = np.meshgrid(kx,ky)
r,theta = cart2pol(kxx,kyy)
rs = np.fft.fftshift(r)
ris = np.arange(0,13.77,0.08)
rim = (ris[1:] + ris[:-1])/2
      
kw = np.zeros((172,499))
with h5py.File('enert_hdf', 'r') as f:
    grup = f.get('ft_total')
    datos = grup['transformada']
    t1 = time()
    for n in range(200):
        df_ft = np.abs( np.fft.fftshift(datos[:,:,n]) )
#        print(df_ft)
        vmr = []
        for i in range(1,len(ris)):
            r1 = np.logical_and(rs > ris[i-1], rs < ris[i])
            mr_dp = np.mean( (np.log(df_ft)[r1>0])**2 )
#            print(mr_dp)
            vmr.append(mr_dp) 
        kw[:,n] = vmr
    t2 = time()
    print(t2-t1)
      
#plt.figure()
#ext = np.array([np.min(w)/1000,np.max(w)/1000,np.min(rim),np.max(rim)])
#plt.imshow(kw, origin='lower', extent=ext)
#plt.show()
#%%
plt.figure()
ext = np.array([np.min(w)/100,np.max(w)/100,np.min(rim),np.max(rim)])
plt.imshow(kw, origin='lower', extent=ext)
#plt.plot(rim,kw[:,])
plt.show()
#%%
#pruebo algo de hdf5
with h5py.File('prueba','w') as f:
    gru = f.create_group('grupi')
    dats = gru.create_dataset('llenar',(10,10,10), dtype='complex')
    dats[0] = np.ones((10,10)) + 1j * np.ones((10,10))
#%%
with h5py.File('ener_hdf','r') as f:
    gr = f.get('espacial')
    grr = gr['llenar'][:,0,0]
    print(np.shape(grr))
    
#%%
#----------------------------------------------------------------------------------
#%%
# Lo que hicimos con pablo para ver si funciona igual
import h5py
import numpy as np
from time import time
#%%
print('creando random set')
dphs = 1j*np.random.randn(1000,1000,100)
# lo hice de 1000,1000,100 porque sino me llenaba la ram
#%%
print('creando archivo con datos')
with h5py.File('archivo_input.hdf5', 'w') as f:
    h_im = f.create_group('df')
    dset = h_im.create_dataset('dp', (1000,1000,500), dtype='complex')
    for i in range(5):
        dset[:,:,100*i:100*i+100] = 1j*np.random.randn(1000,1000,100)
#%%
t1 = time()
with h5py.File('archivo_destino.hdf5', 'w') as f:
    h_im2 = f.create_group('df')
    print('calculando fft temporal')
    destino = h_im2.create_dataset('dp', (1000,1000,500), dtype='complex')
    with h5py.File('archivo_input.hdf5','r') as fr:
            ftesp = fr.get('df')
            st = ftesp['dp']#[ii, jj, :]
            for ii in range(50):
                # print(str(ii) + "====================================")
                for jj in range(1000):
                    # print(jj)
                    st = ftesp['dp'][ii, jj, :]
                    ftij = np.fft.fft(st)   
                    # ftij = np.fft.fft(st[ii,jj,:])   
                    destino[ii, jj, :] = ftij
t2 = time()
print(t2-t1)
# un timeit tal y como esta tarda 1.19 s 
#%%

