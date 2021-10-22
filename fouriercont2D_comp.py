# Fourier Continuation 2D

def fouriercont2D_comp(imref, Mx, My, bx0, by0, D, U, S, V):
    # % imref:        image to complete(nans on holes)
    # % Mx:           modes on x
    # % My:           modes on y
    # % bx0:          bx0*lenx = full size on x direction (periodica)
    # % by0:          by0*leny = full size on y direction (periodica)
    # % D,U,S,V       output of generarD.m
    import numpy as np

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



# Dtyy=D'*yy;
#
# a = V*(diag(diag(S).^-1)*(U.'*Dtyy));
#
# AA=zeros(size(xmesh(:)));
# % disp('Calculando la matriz continuada...')
# % tic
#
# % CONSULTAR CON FELIPE:
# % --> Por que este for va hasta k-1 y no hasta k?
# %
# % for i=1:k-1   % era k-1
# %    AA=AA+DD(:,i).*a(i);
# % end
# % AA = DD(:,1:end-1)*a(1:end-1);
# % Si el loop anterior fuese hasta k:
# AA = DD*a;
#
# AAA=reshape(AA,size(xmesh));
# % disp('Terminado.')
#
# end
#
#
