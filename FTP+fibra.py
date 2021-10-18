import numpy as np
from scipy import signal
import scipy.fft as scf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# The following line requires "unwrap" to be installed (this is a very robust 2D unwrapper)
# the command to install it from pip is simply: "pip install unwrap" 
#from unwrap import unwrap
from skimage.restoration import unwrap_phase as unwrap
from skimage.io import imread
import Func_Splines as spl
#%%
def dphase_2d(dY0,dY,thx,thy,ns,inde=9):
    ny, nx = np.shape(dY)
    
    fY0=np.fft.fft2(dY0)
    fY=np.fft.fft2(dY)
    
#    a = np.abs(fY0[inde:int(np.floor(nx/2)),:])
#    imax = np.unravel_index(np.argmax(a, axis=None), a.shape)
    imax = np.argmax(np.abs(fY0[0,inde:int(np.floor(nx/2))]))
    ifmax_x, ifmax_y = imax+inde, 0
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
    return dp, fY0, gaussfilt1D
#%%
gris = np.zeros((1024,1024))
for i in range(1,301):
    ni = '{:04d}'.format(i)
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\Grises\ID_0_C1S000600'+ni+'.tif')
    ima = np.array(ima,dtype='float64')
    gris += ima
gris = gris/300

ref = np.zeros((1024,1024))
for i in range(1,301):
    ni = '{:04d}'.format(i)
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\Referencia\ID_0_C1S000300'+ni+'.tif')
    ima = np.array(ima,dtype='float64')
    ref += ima
ref = ref/300
#%%
plt.figure()
plt.imshow(gris,cmap='gray')
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(ref,cmap='gray')
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(ref-gris,cmap='gray')
plt.colorbar()
plt.show()
#%%
#en FTP_fibra hay 3072 imagenes
num = 135
ni = '{:04d}'.format(num)
im = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\FTP_fibra\ID_0_C1S000500'+ni+'.tif')

plt.figure()
plt.imshow(im,cmap='gray')
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(im-gris,cmap='gray')
plt.colorbar()
plt.show()
#%%
#FTP, donde esta la fibra se ve medio raro
thx,thy, ns = 0.5, 80, 0.5
dp, fY0, gf = dphase_2d(im-gris, ref-gris,thx,thy,ns,inde=9)

lim = 10
plt.figure()
plt.imshow(dp[lim:-lim,lim:-lim])
plt.title('thx={}, thy={}, ns={}'.format(thx,thy,ns))
plt.colorbar()
plt.show()
#%%
# Reconocer la fibra, es dificil poder ailar la fibra
from skimage.morphology import thin, skeletonize, remove_small_objects, dilation, binary_dilation
from skimage.measure import label, regionprops
from scipy.interpolate import splev, splrep, splprep

bina = -185
#ima = spl.encuentra_fibra([im], connec=4, binariza=bina, eccen=0.999)
ima = (im-gris)<bina
fibra = remove_small_objects(ima, connectivity=4)
li = label(fibra)
props = regionprops(li)
lfs = []
for i in range(np.max(li)):
    ar = props[i].area
    ec = props[i].eccentricity
    if ar < 350 and ar > 150 and ec <= 0.999:
        lf = i+1
        lfs.append(lf)
fib = li==lf
fib = skeletonize(binary_dilation(fib),method='lee') 
fib = np.asarray(fib>0,dtype='uint8')
pro = regionprops(fib)
bb = pro[0].bbox
fibra = fib[bb[0]:bb[2], bb[1]:bb[3]]

plt.figure()
plt.imshow(ima)
plt.show()
#%%

tramos,bordes = spl.cortar_fibra_rap(fibra,bb,cortar_ruido=False)
tramos = spl.ordenar_fibra(tramos)
try:
    curv,spline = spl.pegar_fibra(tramos,bordes,window=17,s=10)
except UnboundLocalError:
    print('\n',i)


t_spl = np.linspace(0, 1, 10000)
xf, yf = splev(t_spl, spline)


plt.figure()
plt.set_cmap('gray')
plt.imshow(im-gris)
plt.plot(xf, yf, 'r-')
plt.show()

#%%
#from skimage.filters.rank import entropy, gradient, majority, noise_filter
from skimage.filters import roberts
#from skimage.morphology import disk, ball
a = im - gris
#a = np.asarray(a,dtype='uint8')
#ac = roberts(a)

plt.figure()
plt.imshow(a)
plt.show()

lin = 370
lina = a[lin,:]
ft= np.fft.fft(lina)
fr = np.fft.fftfreq(lina.size)
ft = ft * (ft>0)

plt.figure()
plt.plot(lina)
plt.show()


