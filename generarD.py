def generarD(imref,Mx,My,bx0,by0):
    import numpy as np
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


# DtD=D'*D;
# [U ,S ,V]=svd(DtD);
# end
