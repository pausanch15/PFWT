#Librerías
import numpy as np
import matplotlib.pyplot as plt
import Func_Splines as spl
import Continuacion_Fourier as cf
from skimage.io import imread
from scipy.interpolate import splev
from time import time
import h5py 
from tqdm import tqdm
plt.ion()

#%%
#Traigo los hdf5 con las alturas y hago el histograma de las alturas
num_bins = 1000
binis = np.linspace(-5, 5, num_bins)
Al = np.zeros(num_bins-1)

#with h5py.File('18-12-2021_ftp.hdf5', 'r') as f:
with h5py.File('alturas.hdf5', 'r') as f:
    gdp = f.get('hs')
    h_dp = gdp['alts']
    for i, j in zip(range(1,len(h_dp)), tqdm(range(1,len(h_dp)))):
        hs = (h_dp[i][10:-10, 10:-10]).flatten()
        his, bins = np.histogram(hs, bins=binis)
        Al += his
#%%
norm = np.sum(Al)
plt.figure()
plt.bar(binis[:-1], height=Al/norm, width=binis[1]-binis[0], edgecolor='black')
plt.grid()
plt.title('Histograma de Alturas')
plt.show()

#%%
#Ahora el de las velocidades
num_bins = 1000
binis = np.linspace(-40, 40, num_bins)
Ve = np.zeros(num_bins-1)

#with h5py.File('18-12-2021_ftp.hdf5', 'r') as f:
with h5py.File('alturas.hdf5', 'r') as f:
    gdp = f.get('hs')
    h_dp = gdp['alts']
    dp0 = h_dp[0][10:-10,10:-10]
    for i, j in zip(range(1,len(h_dp)), tqdm(range(1,len(h_dp)))):
        dp1 = h_dp[i][10:-10,10:-10]
#        dp0 = h_dp[i-1][10:-10,10:-10]
        vel = (250*(dp1-dp0)).flatten()
        his, bins = np.histogram(vel,bins=binis)
        Ve += his
        dp0 = dp1
#%%
norm = np.sum(Ve)
plt.figure()
plt.bar(binis[:-1],Ve/norm,width=binis[1]-binis[0],edgecolor='black')
plt.show()
plt.grid()
plt.title('Histograma de Velocidades')
plt.show()
#%%
#Ahora con las aceleraciones
#a = np.zeros(v_un_frame*n_frames)
num_bins = 1000
binis = np.linspace(-7000, 7000, num_bins)
Ac = np.zeros(num_bins-1)

#with h5py.File('18-12-2021_ftp.hdf5', 'r') as f:
with h5py.File('alturas.hdf5', 'r') as f:
    gdp = f.get('hs')
    h_dp = gdp['alts']
    dp0 = h_dp[0][10:-10,10:-10]
    dp1 = h_dp[1][10:-10,10:-10]
    for i, j in zip(range(1,len(h_dp)-1), tqdm(range(1,len(h_dp)-1))):
#        dp1 = h_dp[i][10:-10,10:-10]
#        dp0 = h_dp[i-1][10:-10,10:-10]
        dp2 = h_dp[i+1][10:-10,10:-10]
        aceleracion = (250**2)*(dp2-2*dp1+dp0)
        aceleracion = aceleracion.flatten()
        his, bins = np.histogram(aceleracion, bins=binis)
        Ac += his
        dp0 = dp1
        dp1 = dp2
#        print( np.mean(aceleracion), np.max(aceleracion), np.min(aceleracion))
#        del(aceleracion, dp1, dp0, dp2)
#%%
#Hago el histograma de aceleración
norm = np.sum(Ac)
plt.figure()
plt.bar(binis[:-1],Ac/norm,width=binis[1]-binis[0],edgecolor='black')
plt.grid()
plt.title('Histograma de Aceleración')
plt.show()
#%%
