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
n = 51
dpfts = np.abs(np.fft.fftshift(dpft[n,:,:]))
rs = np.fft.fftshift(r)
ris = np.arange(0,0.363,0.003)

vmr = []
for i in range(1,len(ris)):
    r1 = np.logical_and(rs > ris[i-1], rs < ris[i])
    mr_dp = np.mean( (np.log(dpfts)[r1>0])**2 )
    vmr.append(mr_dp) 

rim = (ris[1:]+ris[:-1])/2     

plt.figure()
plt.plot(rim,vmr)
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
#499 = len(dps)
#1004 para sacar 10 pixeles de cada borde
ft_espacial = np.zeros((1004,1004)) + 1j * np.zeros((1004,1004))
with h5py.File('ener_hdf', 'w') as f:
    esp_ft = f.create_group('espacial')
    for i in range(499):
        if i%20 == 0: print(i,end=' ')
        ftes = np.fft.fft2(dps[i][10:-10,10:-10]) 
        ft_espacial = np.fft.fftshift(ftes)
        esp_ft.create_dataset('tiem_'+str(i),data=ft_espacial)
#%%
#with h5py.File('enert_hdf', 'w') as f:
#    
#ft_esp_tem = np.zeros((499,1004,1004))
#%%
n = 170
plt.figure()
plt.imshow(np.log(ft_espacial[n]))
plt.colorbar()
plt.show()
#%%
#pruebo algo de hdf5
with h5py.File('prueba','w') as f:
    gru = f.create_group('grupi')
    dats = gru.create_dataset('llenar',(10,10,10), dtype='complex')
    dats[0] = np.ones((10,10)) + 1j * np.ones((10,10))
#%%
with h5py.File('prueba','r') as f:
    gr = f.get('grupi')
    grr = gr['llenar']
    print(grr[0])


