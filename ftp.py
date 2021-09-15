import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# The following line requires "unwrap" to be installed (this is a very robust 2D unwrapper)
# the command to install it from pip is simply: "pip install unwrap" 
#from unwrap import unwrap
from skimage.restoration import unwrap_phase as unwrap
from skimage.io import imread
#%%
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
if __name__ == "__main__":
    plt.close('all')
    # demo:
    # - create an arbitrary phase map
    v = np.linspace(-3, 3, 512)
    x, y = np.meshgrid(v, v)
    phase_imposed =  (3*(1-x)**2.*np.exp(-(x**2) - (y+1)**2) - 10*(x/5 - x**3 - y**5)*np.exp(-x**2-y**2) - 1/3*np.exp(-(x+1)**2 - y**2))

    # - build reference & deformed images
    # - apply ftp to recover height map
    f0 = 40
    im_ref = np.sin(f0*x)
    im_def = np.sin(f0*x + phase_imposed)

    # we use the function defined above
    recovered_phase_map = calculate_phase_diff_map_1D(im_def, im_ref, 0.5, 0.2)
    # we know that at (0,0) the phase map difference should be zero, so
    recovered_phase_map = recovered_phase_map - recovered_phase_map[0,0] 

    # results
    plt.figure()
    plt.imshow(phase_imposed)
    plt.title("phase imposed (ground truth)")
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.imshow(im_ref)
    plt.title("reference image")
    plt.show()

    plt.figure()
    plt.imshow(im_def)
    plt.title("deformed image")
    plt.show()

    plt.figure()
    plt.title("recovered phase map")
    plt.imshow(recovered_phase_map)
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.title("error = abs( recovered phase map - phase imposed)")
    plt.imshow(np.abs(recovered_phase_map - phase_imposed))
    plt.colorbar()
    plt.show()
#%%
im_p = np.zeros((1024,1024))
for i in range(1,301):
    ni = '{:04d}'.format(i)
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\ID_0_C1S0003\ID_0_C1S000300'+ni+'.tif')
    im_p += ima
im_p = im_p/300
#%%
#im_p = np.zeros((1024,1024))
#for i in range(1,301):
#    ni = '{:04d}'.format(i)
#    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\ID_0_C1S0002\ID_0_C1S000200'+ni+'.tif')
#    im_p += ima
#im_p = im_p/300    
i = 5
ni = '{:04d}'.format(i)
imr = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\ID_0_C1S0002\ID_0_C1S000200'+ni+'.tif')
j = 1504
ni2 = '{:04d}'.format(j)
im = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\ID_0_C1S0001\ID_0_C1S000100'+ni2+'.tif')

plt.figure()
plt.imshow(im-im_p,cmap='gray')
plt.show()
plt.figure()
plt.imshow(im,cmap='gray')
plt.show()
#plt.figure()
#plt.imshow(imr-im_p,cmap='gray')
#plt.show()
#%%
recovered_phase_map = calculate_phase_diff_map_1D(im-im_p, imr-im_p, 0.2, 0.2)
recovered_phase_map = recovered_phase_map - recovered_phase_map[0,0] 
plt.figure()
plt.title("recovered phase map")
plt.imshow(recovered_phase_map)
plt.colorbar()
plt.show()

recovered_phase_map = calculate_phase_diff_map_1D_2(im-im_p, imr-im_p, 0.2, 0.2)
recovered_phase_map = recovered_phase_map - recovered_phase_map[0,0] 
plt.figure()
plt.title("recovered phase map")
plt.imshow(recovered_phase_map)
plt.colorbar()
plt.show()
#
#recovered_phase_map = calculate_phase_diff_map_1D(im-im_p, imr-im_p, 0.3, 0.3)
#recovered_phase_map = recovered_phase_map - recovered_phase_map[0,0] 
#plt.figure()
#plt.title("recovered phase map")
#plt.imshow(recovered_phase_map)
#plt.colorbar()
#plt.show()
#%%
def calculate_phase_diff_map_1D_2(dY, dY0, th, ns):
    ny, nx = np.shape(dY)
    phase0 = np.zeros([nx,ny])
    phase  = np.zeros([nx,ny])

    dfy=1./ny
    fy=np.arange(dfy,1,dfy)

    fY0 = np.fft.fft2(dY0)

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

        phase0[lin,:] = phase0[lin,:] - phase0[lin,0] + phase0[0,0]
        phase[lin,:]  = phase[lin,:] - phase[lin,0] + phase[0,0] 
    
    mphase0 = unwrap(phase0)
    mphase  = unwrap(phase)
    
    # Definition of the phase difference map
    dphase = (mphase-mphase0)
    return dphase
#%%
def dphase_2d(dY0,dY,thx,thy,ns):
    ny, nx = np.shape(dY)
    
    fY0=np.fft.fft2(dY0)
    fY=np.fft.fft2(dY)
    
    inde = 40
    a = np.abs(fY0[inde:int(np.floor(nx/2)),inde:int(np.floor(ny/2))])
    imax = np.unravel_index(np.argmax(a, axis=None), a.shape)
    ifmax_x, ifmax_y = imax[0]+inde, imax[1]+inde
    print(ifmax_x, ifmax_y)
                           
    HW_x, HW_y = np.round(ifmax_x*thx), np.round(ifmax_y*thy)
    W_x, W_y = 2*HW_x, 2*HW_y
    win_x, win_y = signal.tukey(int(W_x),ns), signal.tukey(int(W_y),ns)
    wxx, wyy = np.meshgrid(win_x,win_y)
    win = wxx * wyy
    
    gaussfilt1D = np.zeros((nx,ny))
    gaussfilt1D[int(ifmax_y-HW_y-1):int(ifmax_y-HW_y+W_y-1),
                         int(ifmax_x-HW_x-1):int(ifmax_x-HW_x+W_x-1)] = win
    
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
    
    dp = mphase - mphase0
    dp = dp - dp[0,0]
    return dp, gaussfilt1D, fY0
#%%
dp, gafi, fY0 = dphase_2d(im-im_p, imr-im_p,0.2,0.2,0.2)
bx,by = 512, 256
lx,ly = 10,5
x, y = np.linspace(bx-lx,bx+lx,2*lx+1), np.linspace(by-ly,by+ly,2*ly+1)
X,Y = np.meshgrid(y,x)

fyt = np.abs(fY0[bx-lx:bx+lx+1,by-ly:by+ly+1])
#
plt.figure()
plt.imshow(gafi)
plt.colorbar()
plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, fyt)
plt.show()
plt.figure()
plt.imshow(dp)
plt.colorbar()
plt.show()
#%%
dp,a,b = dphase_2d(sato(im-im_p,mode='reflect'), sato(imr-im_p,mode='reflect'),0.2,0.2,0.2)
plt.figure()
plt.imshow(dp)
plt.colorbar()
plt.show()

dp = dphase_2d(im-im_p, imr-im_p,0.02,0.2)
plt.figure()
plt.imshow(dp)
plt.colorbar()
plt.show()

dp = dphase_2d(im-im_p, imr-im_p,0.2,2)
plt.figure()
plt.imshow(dp)
plt.colorbar()
plt.show()
#%%
from skimage.filters import sato
A = sato(im-im_p,mode='reflect')
fa = np.fft.fft2(A)
ifa = np.fft.ifft(fa)
ass = np.angle(ifa)

x = np.linspace(0,1023,1024)
X,Y = np.meshgrid(x,x)

plt.figure()
plt.imshow(A)
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(ass)
plt.colorbar()
plt.show()
plt.figure()
plt.plot(fa[125,:])
plt.show()
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.plot_surface(X, Y, fa)
#plt.show()
