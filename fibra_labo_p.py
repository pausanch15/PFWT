#Trato de levantar las imágenes del labo.
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splev, splrep, splprep
from skimage.morphology import thin, skeletonize, remove_small_objects, binary_dilation, dilation
from skimage.measure import label, regionprops
from skimage.io import imread
import Func_Genera_Fibras_2 as gf
import Func_Splines as spl
from time import time
from scipy import interpolate
import scipy.stats as sps
import h5py 
from PIL import Image
plt.ion()

#%%
#Carpeta con las imágenes que voy a probar: ID_0_C1S0001
#Las imágenes se llaman ID_0_C1S000100i, con i entre 0001 y 3072
#convert original.tiff -compress none -type palette -depth 16 nuevo.tiff
imread('/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0001/ID_0_C1S0001000001_16.tif')

#%%
#Acá trato de ver los de ftp
import numpy as np
from scipy import signal
import scipy.fft as scf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from unwrap import unwrap
from skimage.restoration import unwrap_phase as unwrap
from skimage.io import imread
plt.ion()

#%%
#Lo que nos dio Pablo
def calculate_phase_diff_map_1D(dY, dY0, th, ns):
    """
    # % Basic FTP treatment.
    # % This function takes a deformed and a reference image and calculates the phase difference map between the two.
    # %
    # % INPUTS:
    # % dY	= deformed image
    # % dY0	= reference image
    # % th  = factor for tukey window size
    # % ns	= size of gaussian filter
    # %
    # % OUTPUT:
    # % dphase 	= phase difference map between images
    """

    ny, nx = np.shape(dY)
    phase0 = np.zeros([nx,ny])
    phase  = np.zeros([nx,ny])

    dfy=1./ny
    fy=np.arange(dfy,1,dfy)

    for lin in range (0,nx):
        fY0=np.fft.fft(dY0[lin,:])
        fY=np.fft.fft(dY[lin,:])

        imax=np.argmax(np.abs(fY0[9:int(np.floor(nx/2))]))
        ifmax=imax+9

        HW=np.round(ifmax*th)
        W=2*HW
        win=signal.tukey(int(W),ns)

        gaussfilt1D = np.zeros(nx)
        gaussfilt1D[int(ifmax-HW-1):int(ifmax-HW+W-1)] = win

        # Multiplication by the filter
        Nfy0 = fY0*gaussfilt1D
        Nfy = fY*gaussfilt1D

        # Inverse Fourier transform of both lines
        Ny0 = np.fft.ifft(Nfy0)
        Ny  = np.fft.ifft(Nfy)
 
        phase0[lin,:] = np.angle(Ny0)
        phase[lin,:]  = np.angle(Ny)
    
    mphase0 = unwrap(phase0)
    mphase  = unwrap(phase)
    
    # Definition of the phase difference map
    dphase = (mphase-mphase0)
    return dphase

#%%
#Lo que hizo Tomi
def dphase_2d(dY0 ,dY, thx, thy, ns):
    """
    # % Basic FTP treatment.
    # % This function takes a deformed and a reference image and calculates the phase difference map between the two.
    # %
    # % INPUTS:
    # % dY	= deformed image
    # % dY0	= reference image
    # % thx = factor for tukey window size in x
    # % thy = factor for tukey window size in y
    # % ns	= size of gaussian filter
    # %
    # % OUTPUT:
    # % dp = phase difference map between images
    # % gaussfilt1D = el filtro usado?
    # % fY0 = transformada de Fourier de la imagen de referencia?
    """
    ny, nx = np.shape(dY)
    
    fY0=np.fft.fft2(dY0)
    fY=np.fft.fft2(dY)
    
    inde = 9 #Qué es esto?
    a = np.abs(fY0[inde:int(np.floor(nx/2)),:])
    imax = np.unravel_index(np.argmax(a, axis=None), a.shape)
    ifmax_x, ifmax_y = imax[1]+inde, 0
    print(ifmax_x, ifmax_y)
                           
    HW_x, HW_y = np.round(ifmax_x*thx), np.round(thy)
    W_x, W_y = 2*HW_x, 2*HW_y
    win_x, win_y = signal.tukey(int(W_x),ns), signal.tukey(int(W_y),ns)
    wxx, wyy = np.meshgrid(win_x,win_y)
    win = wxx * wyy
    
    
    gaussfilt1D = np.zeros((nx,ny))
    gaussfilt1D[int(ifmax_y-HW_y):,
                int(ifmax_x-HW_x-1):int(ifmax_x-HW_x+W_x-1)] = win[:int(ifmax_y-HW_y+W_y),:]
    gaussfilt1D[:int(ifmax_y-HW_y+W_y),
                int(ifmax_x-HW_x-1):int(ifmax_x-HW_x+W_x-1)] = win[int(ifmax_y-HW_y):,:]
    
    # Multiplication by the filter
    Nfy0 = fY0*gaussfilt1D
    Nfy = fY*gaussfilt1D
    
    # Inverse Fourier transform of both lines
    Ny0 = np.fft.ifft2(Nfy0)
    Ny  = np.fft.ifft2(Nfy)
    # 
    phase0 = np.angle(Ny0)
    phase = np.angle(Ny)
    
    mphase0 = unwrap(phase0)
    mphase  = unwrap(phase)
    
    dp = mphase0 - mphase
    dp = dp - dp[0,0]
    return dp, gaussfilt1D, fY0

#%%
#Probamos esto
#Trigo las imágenes y las veo
imr = imread('/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0002/ID_0_C1S0002000005_16.tif')
im = imread('/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0001/ID_0_C1S0001000005_16.tif')

plt.figure()
plt.imshow(im, cmap='gray')
plt.title('Imagen: im')
plt.show()

plt.figure()
plt.imshow(imr,cmap='gray')
plt.title('Imagen: imr')
plt.show()

#Falta crear la imagen promedio de grises, pero necesito convertir todas las imagenes a 16 bits y no se hacer ese loop.
