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
num_bins = 10000
binis = np.linspace(-num_bins/2, num_bins/2, num_bins)
H = np.zeros(num_bins-1)

with h5py.File('18-12-2021_ftp.hdf5', 'r') as f:
    gdp = f.get('hs')
    h_dp = gdp['alts']
    for i, j in zip(range(1,len(h_dp)), tqdm(range(1,len(h_dp)))):
        hs = (h_dp[i][10:-10, 10:-10]).flatten()
        his, bins = np.histogram(hs, bins=binis)
        H += his

norm = np.sum(H)
plt.figure()
plt.bar(binis[:-1], height=H/norm, width=binis[1]-binis[0], edgecolor='black')
plt.grid()
plt.title('Histograma de Alturas')
plt.show()

#%%
#Ahora el de las alturas
with h5py.File(ftp_hdf, 'r') as f:
    gdp = f.get('dif_phase')
    h_dp = gdp['dp']
    for i, j in zip(range(1,len(h_dp)), tqdm(range(1,len(h_dp)))):
        dp1 = h_dp[i][10:-10,10:-10]
        dp0 = h_dp[i-1][10:-10,10:-10]
        vel = (250*(dp1-dp0)).flatten()
        his, bins = np.histogram(vel,bins=binis)
        H += his

norm = np.sum(H)
plt.figure()
plt.bar(binis[:-1],H/norm,width=binis[1]-binis[0],edgecolor='black')
plt.show()
plt.grid()
plt.title('Histograma de Velocidades')
plt.show()

#Ahora con las aceleraciones
a = np.zeros(v_un_frame*n_frames)

with h5py.File(ftp_hdf, 'r') as f:
    gdp = f.get('dif_phase')
    h_dp = gdp['dp']
    for i, j in zip(range(1,len(h_dp)), tqdm(range(1,len(h_dp)))):
        dp1 = h_dp[i][10:-10,10:-10]
        dp0 = h_dp[i-1][10:-10,10:-10]
        dp2 = h_dp[i+1][10:-10,10:-10]
        aceleracion = (250**2)*(dp2-2*dp0+dp1)
        aceleracion = aceleracion.flatten()
        his, bins = np.histogram(aceleracion, bins=binis)
        del(aceleracion, dp1, dp0, dp2)

#Hago el histograma de aceleración
norm = np.sum(H)
plt.figure()
plt.bar(binis[:-1],H/norm,width=binis[1]-binis[0],edgecolor='black')
plt.grid()
plt.title('Histograma de Aceleración')
plt.show()
