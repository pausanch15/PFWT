import numpy as np
import matplotlib.pyplot as plt
import h5py 
#%%
dps = []
ftp_hdf = r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\Analisado\ftp.hdf5'
with h5py.File(ftp_hdf, 'r') as f:
    gdp = f.get('dif_phase')
    h_dp = gdp['dp']
    for i in range(len(h_dp)):
        dps.append(h_dp[i])
#%%
plt.figure()
plt.imshow(dps[0][300:900,300:900])
plt.show()
#%%
nt, nxy = 200, 600
dps_c = np.zeros((nt,nxy,nxy))
for i in range(nt):
    dps_c[i] = dps[i][300:900,300:900] - np.mean(dps[i][300:900,300:900])
#%%
dpft = np.fft.fftn(dps_c)
w, kx, ky = np.fft.fftfreq(nt,1/250),0.159* np.fft.fftfreq(nxy,0.2106), 0.18*np.fft.fftfreq(nxy,0.2106) 
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
#%%
n = 3
#ext = 2*np.pi*np.array([np.min(kx),np.max(kx),np.min(ky),np.max(ky)])
ext = 2*np.pi * np.array([np.min(kx),np.max(kx),np.min(w),np.max(w)])
fig = plt.figure()
plt.imshow(np.log(np.abs(np.fft.fftshift(dpft[:,:,n])))**2, extent=ext,aspect='auto')
#plt.plot(kx[:150]*2*np.pi,ws,'r-')
plt.plot(-kx[:150],-ws,'r--')
plt.plot(-kx[:150],ws,'r--')
plt.plot(kx[:150],-ws,'r--')
plt.plot(kx[:150],ws,'r--')
plt.colorbar()
plt.xlabel('kx')
plt.ylabel('w')
plt.ylim([-780,780])
plt.show()
fig = plt.figure()
plt.imshow(np.log(np.abs(np.fft.fftshift(dpft[:,:,n])))**2, extent=ext,aspect='auto')
plt.colorbar()
plt.xlabel('kx')
plt.ylabel('w')
plt.ylim([-780,780])
plt.show()

#%%

#%%
plt.figure()
lin = np.array(kx[:100])**-1 * 10
plt.loglog(kx[:100], np.abs(dpft[0,:100,0]))
plt.loglog(kx[:100], lin)
plt.grid()
plt.show()
