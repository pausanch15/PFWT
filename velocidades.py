#Intento calcular las velocidades que dijo Pablo
#%%
#Librerías
import numpy as np
import matplotlib.pyplot as plt
import Func_Splines as spl
import Continuacion_Fourier as cf
from skimage.io import imread
from scipy.interpolate import splev
import h5py 
plt.ion()

#%%
#Traigo los hdf5 de las imagenes de ftp
#Si traigo los 250 que hice se me tilda la pc, así que hago todo para 50
dps = []

ftp_hdf = r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ftp2.hdf5'

with h5py.File(ftp_hdf, 'r') as f:
    gdp = f.get('dif_phase')
    h_dp = gdp['dp']
    # for i in range(len(h_dp)):
        # dps.append(h_dp[i])
    for i in range(50):
        dps.append(h_dp[i])

#%%
#Calculo la velocidad
v_un_frame = np.shape((dps[1] - dps[0]).flatten())[0] #Cantidad de velocidades por frame
n_frames = len(dps)
v = np.zeros(v_un_frame*n_frames) #array que va a tener la velocidad de cada punto de todos los frames

for i, im in enumerate(dps):
    i = i + 1
    if i == len(dps):
        break
    velocidad = (250*(dps[i]-im)).flatten()
    # velocidad = list(velocidad)
    inic =  (i-1)*v_un_frame
    fin = i*v_un_frame
    v[inic:fin] = velocidad
    del(velocidad, inic, fin)

#%%
#Hago el histograma de velocidad
plt.hist(v, bins=100, density=True, edgecolor="black")
plt.grid()
plt.title('Histograma de Velocidades')
plt.show()

#%%
#Ahora la aceleración
a = np.zeros(v_un_frame*n_frames)

for i in range(1, len(dps)-1):
    aceleracion = (250**2)*(dps[i+1]-2*dps[i]+dps[i-1])
    aceleracion = aceleracion.flatten()
    inic =  (i-1)*v_un_frame
    fin = i*v_un_frame
    a[inic:fin] = aceleracion
    del(aceleracion, inic, fin)

#%%
#Hago el histograma de aceleración
plt.hist(v, bins=100, density=True, edgecolor="black")
plt.grid()
plt.title('Histograma de Aceleración')
plt.show()
