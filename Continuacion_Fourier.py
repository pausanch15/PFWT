#Las funciones para hacer la continuación de Fourier
#Tambien esta la funcion para hacer FTP en 2d
import numpy as np
from scipy import signal
from skimage.restoration import unwrap_phase as unwrap

def fouriercont2D_comp(imref, Mx, My, bx0, by0, D, U, S, V):
    # % imref:        image to complete(nans on holes)
    # % Mx:           modes on x
    # % My:           modes on y
    # % bx0:          bx0*lenx = full size on x direction (periodica)
    # % by0:          by0*leny = full size on y direction (periodica)
    # % D,U,S,V       output of generarD.m

    d1,d2=np.shape(imref)

    bx=np.floor(bx0*(d2-1))
    by=np.floor(bx0*(d1-1))

    x=np.arange(bx)
    y=np.arange(by)

    xmesh, ymesh=np.meshgrid(x,y)

    xx=xmesh.flatten(order='F')
    yy=ymesh.flatten(order='F')

    k=0
    DD=np.zeros([int(bx*by),int(4*Mx*My)])
    Hxcos=np.zeros([int(bx*by),int(Mx)])
    Hycos=np.zeros([int(bx*by),int(My)])
    Hxsin=np.zeros([int(bx*by),int(Mx)])
    Hysin=np.zeros([int(bx*by),int(My)])

    for i in range (0,Mx):
        Hxcos[:,i]=np.cos(2*np.pi/bx*i*xx)
        Hxsin[:,i]=np.sin(2*np.pi/bx*(i+1)*xx)
    for j in range (0,My):
        Hycos[:,j]=np.cos(2*np.pi/by*j*yy)
        Hysin[:,j]=np.sin(2*np.pi/by*(j+1)*yy)
    for ii in range (0,Mx):
        for jj in range (0,My):
            DD[:,k]=Hxcos[:,ii]*Hycos[:,jj]
            k=k+1
    for ii in range (0,Mx):
        for jj in range (0,My):
            DD[:,k]=Hxsin[:,ii]*Hysin[:,jj]
            k=k+1
    for ii in range (0,Mx):
        for jj in range (0,My):
            DD[:,k]=Hxcos[:,ii]*Hysin[:,jj]
            k=k+1
    for ii in range (0,Mx):
        for jj in range (0,My):
            DD[:,k]=Hxsin[:,ii]*Hycos[:,jj]
            k=k+1

    yyy =imref.flatten(order='F')
    yyy=yyy[~np.isnan(yyy)]
    Dtyy=np.dot(D.T,yyy)
    a = V.T.dot(np.diag(S**-1)).dot(U.T).dot(Dtyy)

    AA=np.zeros(np.shape(xmesh.flatten(order='F')))
    AA=np.dot(DD,a)
    AAA=np.reshape(AA,np.shape(xmesh),order='F')
    return AAA


def generarD(imref,Mx,My,bx0,by0):
    d1,d2=np.shape(imref)

    x=np.arange(d2)
    y=np.arange(d1)

    xmesh, ymesh=np.meshgrid(x,y)

    bx=np.floor(bx0*x[-1])
    by=np.floor(bx0*y[-1])

    xx=xmesh.flatten(order='F')
    yy=ymesh.flatten(order='F')

    k=0
    D=np.zeros([int(d1*d2),int(4*Mx*My)])
    Hxcos=np.zeros([int(d1*d2),int(Mx)])
    Hycos=np.zeros([int(d1*d2),int(My)])
    Hxsin=np.zeros([int(d1*d2),int(Mx)])
    Hysin=np.zeros([int(d1*d2),int(My)])

    for i in range (0,Mx):
        Hxcos[:,i]=np.cos(2*np.pi/bx*i*xx)
        Hxsin[:,i]=np.sin(2*np.pi/bx*(i+1)*xx)
    for j in range (0,My):
        Hycos[:,j]=np.cos(2*np.pi/by*j*yy)
        Hysin[:,j]=np.sin(2*np.pi/by*(j+1)*yy)
    for ii in range (0,Mx):
        for jj in range (0,My):
            D[:,k]=Hxcos[:,ii]*Hycos[:,jj]
            k=k+1
    for ii in range (0,Mx):
        for jj in range (0,My):
            D[:,k]=Hxsin[:,ii]*Hysin[:,jj]
            k=k+1
    for ii in range (0,Mx):
        for jj in range (0,My):
            D[:,k]=Hxcos[:,ii]*Hysin[:,jj]
            k=k+1
    for ii in range (0,Mx):
        for jj in range (0,My):
            D[:,k]=Hxsin[:,ii]*Hycos[:,jj]
            k=k+1

    # nuew numpy version
    # D = D[~np.isnan(imref.flatten(1))]
    D = D[~np.isnan(imref.flatten(order='F'))]
    DtD=np.dot(D.T,D)

    U,S,V=np.linalg.svd(DtD)

    return D,U,S,V

def hacer_fou_cont(im,fibra,bb,ventana=150,mmx=2,mmy=2):
    # im tiene que ser float
    # lo hace para ventana cuadrada de tamaño (ventana x ventana)
    bb1,bb2 = int((bb[0]+bb[2])/2), int((bb[1]+bb[3])/2)
    vent = int(ventana/2)
    b1,b2,b3,b4 = bb1-vent,bb1+vent,bb2-vent,bb2+vent
    im_nf = im[bb[0]:bb[2], bb[1]:bb[3]] * (1-fibra)
    im_nf[im_nf==0] = np.nan
    def_hole = np.zeros(np.shape(im))
    def_hole += im
    def_hole[bb[0]:bb[2], bb[1]:bb[3]] = im_nf
    def_hole = def_hole[b1:b2,b3:b4] 
    
    Lx, Ly = np.shape(def_hole)
    d1,d2 = np.shape(def_hole)

    k=np.arange(0,(Lx)/2)/(Lx)
    kxs=np.zeros(Lx)
    for ii in range (0,Lx):
        fftdefx=np.fft.fft(def_hole[ii,:])
        P2=np.abs(fftdefx/Lx)
        # forzamos cero en componente DC
        P2[0] = 0
        P1=P2[0:int(np.floor(Ly/2+1))]
        P1[1:-1]=2*P1[1:-1]
        imax=np.argmax(np.abs(P1))
        kxs[ii]=k[imax]
    
    kx=np.sum(kxs[kxs!=0])/len(kxs[kxs!=0])
    bx0=(.8/kx+d2)/d1
    by0=(.8/kx+d2)/d1
    Mx = int(np.floor(bx0*Lx*2*kx))
    Mx *= mmx
#    print('Mx=' + str(Mx))

    k=np.arange(0,(Ly)/2)/(Ly)
    kys=np.zeros(Ly)    
    for jj in range (0,Ly):
        fftdefy=np.fft.fft(def_hole[:,jj])
        P2=np.abs(fftdefy/Ly)
        P1=P2[0:int(np.floor(Lx/2+1))]
        P1[1:-1]=2*P1[1:-1]
        imax=np.argmax(np.abs(P1))
        imax2=imax
        while P1[imax2]>P1[imax]/10:
                imax2=imax2+1
    #    kys[ii]=k[imax2]
        kys[jj]=k[imax2]    
        
    ky=np.max(kys[kys!=0])    
    My = int(np.floor(by0*Ly*ky)+5)
    My *= mmy
    
    D, U ,S, V = generarD(def_hole,Mx,My,bx0,by0)
    def_hole_FC = fouriercont2D_comp(def_hole,Mx,My,bx0,by0,D,U,S,V)
    
    fib_fc = def_hole_FC[:Lx,:Ly]
    imfc = np.zeros_like(im)
    imfc += im
    imfc[b1:b2,b3:b4] = fib_fc
    return imfc, fib_fc

def dphase_2d(dY0,dY,thx,thy,ns,inde=9):
    ny, nx = np.shape(dY)
    
    fY0=np.fft.fft2(dY0)
    fY=np.fft.fft2(dY)
    
    imax = np.argmax(np.abs(fY0[0,inde:int(np.floor(nx/2))]))
    ifmax_x, ifmax_y = imax+inde, 0
#    print(ifmax_x, ifmax_y)
                           
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
