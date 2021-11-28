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

#ftp_hdf = r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ftp2.hdf5'
ftp_hdf = r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\Analisado\ftp2.hdf5'
# Asi ya queda para que los dos podamos correr mas facil

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
#%%
#-------------------------------------------
#Pruebo cosas
dps = []

#ftp_hdf = r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ftp2.hdf5'
ftp_hdf = r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\Analisado\ftp2.hdf5'

with h5py.File(ftp_hdf, 'r') as f:
    gdp = f.get('dif_phase')
    h_dp = gdp['dp']
    # for i in range(len(h_dp)):
        # dps.append(h_dp[i])
    for i in range(50):
        dps.append(h_dp[i][10:-10,10:-10])
#%%
v_un_frame = np.shape((dps[1] - dps[0]).flatten())[0] #Cantidad de velocidades por frame
n_frames = len(dps)
v = np.zeros(v_un_frame*(n_frames-1)) #array que va a tener la velocidad de cada punto de todos los frames
print(np.shape(v))
for i in range(1,len(dps)):
    if i == len(dps):
        break
    velocidad = (250*(dps[i]-dps[i-1])).flatten()
#    print(i, np.max(velocidad))
    # velocidad = list(velocidad)
    inic =  (i-1)*v_un_frame
    fin = i*v_un_frame
    v[inic:fin] = velocidad
    print(i, inic, fin,  np.max(v[inic:fin]))
    del(velocidad, inic, fin)
#%%
binis = np.linspace(-50,50,100)
his, bins = np.histogram(v,bins=binis)
wi = bins[1]-bins[0]
plt.figure()
plt.bar(bins[:-1],his,width=wi,edgecolor='black')
plt.show()
#%%
n = 48
plt.figure()
#plt.imshow(dps[n]-dps[n-1])
#plt.colorbar()
plt.hist(v, bins=100, density=True, edgecolor="black")
plt.grid()
plt.title('Histograma de Velocidades')
plt.show()
#%% 
# Pruebo con todos los datos
#ftp_hdf = r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ftp2.hdf5'
ftp_hdf = r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\Analisado\ftp2.hdf5'

num_bins = 100
binis = np.linspace(-50,50,num_bins)
H = np.zeros(num_bins-1)

from time import time
t1 = time()
with h5py.File(ftp_hdf, 'r') as f:
    gdp = f.get('dif_phase')
    h_dp = gdp['dp']
    for i in range(1,len(h_dp)):
        if (i-1)%50 == 0: print(i, end=' ')
        dp1 = h_dp[i][10:-10,10:-10]
        dp0 = h_dp[i-1][10:-10,10:-10]
        vel = (250*(dp1-dp0)).flatten()
        his, bins = np.histogram(vel,bins=binis)
        H += his
t2 = time()
print(t2-t1) 
# corrio en un minuto
#%%
norm = np.sum(H)
plt.figure()
plt.bar(binis[:-1],H/norm,width=binis[1]-binis[0],edgecolor='black')
plt.show()
#%%
bin_cen = (binis[1:]+binis[:-1])/2
media = 0
for i in range(99):
    media += H[i]/norm * bin_cen[i]
std2 = 0
for i in range(99):
    std2 += (bin_cen[i] - media)**2 * H[i]/norm
print(media, np.sqrt(std2))
#%%
def normal(x,mu,std):
    mult = 1 / (std * np.sqrt(2*np.pi))
    expo = (x - mu)**2 / (2 * std**2)
    return mult * np.exp(-expo)

dist = normal(bin_cen,media,np.sqrt(std2))

plt.figure()
plt.bar(binis[:-1],H/norm,width=binis[1]-binis[0],edgecolor='black')
plt.plot(bin_cen,dist,'r-')
plt.show()
