import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
# The following line requires "unwrap" to be installed (this is a very robust 2D unwrapper)
# the command to install it from pip is simply: "pip install unwrap" 
from unwrap import unwrap

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
    

    
