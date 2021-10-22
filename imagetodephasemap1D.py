def imagestodphasemap1D(dY,dY0,th,ns):
    # % Basic FTP treatment.
    # % This function takes a deformed and a reference image and calculates the phase difference map between the two.
    # %
    # % INPUTS:
    # % dY	= deformed image
    # % dY0	= reference image
    # % ns	= size of gaussian filter
    # %
    # % OUTPUT:
    # % dphase 	= phase difference map between images
    import numpy as np
    from scipy import signal

    ny,nx = np.shape(dY)
    phase0 = np.zeros([nx,ny])
    phase  = np.zeros([nx,ny])

    for lin in range (0,nx):
        fY0=np.fft.fft(dY0[lin,:])
        fY=np.fft.fft(dY[lin,:])

        dfy=1./ny
        fy=np.arange(dfy,1,dfy)

        imax=np.argmax(np.abs(fY0[9:int(np.floor(nx/2))]))
        ifmax=imax+9

        HW=np.round(ifmax*th)
        W=2*HW
        win=signal.tukey(int(W),ns)

        gaussfilt1D= np.zeros([1,nx])
        gaussfilt1D[0,int(ifmax-HW-1):int(ifmax-HW+W-1)]=win

        # Multiplication by the filter
        Nfy0 = fY0*gaussfilt1D
        Nfy = fY*gaussfilt1D

        # Inverse Fourier transform of both images
        Ny0=np.fft.ifft(Nfy0)
        Ny=np.fft.ifft(Nfy)

        phase0[lin,:]=np.unwrap(np.angle(Ny0))
        phase[lin,:]=np.unwrap(np.angle(Ny))


    # Definition of the phase difference map
    dphase=(phase-phase0);
    return dphase
