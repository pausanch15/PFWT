import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
plt.ion()
# imports que podrian estar en este mismo archivo!
from generarD import generarD
from fouriercont2D_comp import fouriercont2D_comp
# creo que esto no hace falta!
from imagetodephasemap1D import imagestodphasemap1D
#%%
             
# cargamos archivo de medicion -------------------------------------------
imn = np.load('imn.npy')

imn = imn[250:425, 200:375]
def_hole = imn
Lx, Ly = np.shape(def_hole)
#Lx, Ly = np.shape(def_hole)
#Ly, Lx = np.shape(def_hole)
d1,d2 = np.shape(def_hole)
#x,y=np.meshgrid(np.arange(Lx),np.arange(Ly))

plt.figure()
plt.imshow(def_hole)
plt.show()
print(np.shape(imn))
#print('paso carga')
#%%
# Determinacion de Mx y My

k=np.arange(0,(Lx)/2)/(Lx)
kxs=np.zeros(Lx)

for ii in range (0,Lx):
    fftdefx=np.fft.fft(def_hole[ii,:])
    P2=np.abs(fftdefx/Lx)
    # forzamos cero en componente DC
    P2[0] = 0
    P1=P2[0:int(np.floor(Lx/2+1))]
    P1[1:-1]=2*P1[1:-1]
    imax=np.argmax(np.abs(P1))
    kxs[ii]=k[imax]

kx=np.sum(kxs[kxs!=0])/len(kxs[kxs!=0])
bx0=(.8/kx+d2)/d1
by0=(.8/kx+d2)/d1
Mx = int(np.floor(bx0*Lx*2*kx))
Mx *= 2

print('Mx=' + str(Mx))

k=np.arange(0,(Ly)/2)/(Ly)
kys=np.zeros(Ly)

for jj in range (0,Ly):
    fftdefy=np.fft.fft(def_hole[:,jj])
    P2=np.abs(fftdefy/Ly)
    P1=P2[0:int(np.floor(Ly/2+1))]
    P1[1:-1]=2*P1[1:-1]
    imax=np.argmax(np.abs(P1))
    imax2=imax
    while P1[imax2]>P1[imax]/10:
            imax2=imax2+1
    kys[ii]=k[imax2]

ky=np.max(kys[kys!=0])

My = int(np.floor(by0*Ly*ky)+5)
My *= 2

print('My=' + str(My))

if Mx*My>100*10:
    print('****************************')
    print('Matriz para SVD grande, continuar?')
    print(Mx)
    print(My)
    print('****************************')
    lo=input("1 para continuar, cualquier otra tecla para abortar: ")
    if lo!=1:
        import sys
        sys.exit("Abortado")


D, U ,S, V=generarD(def_hole,Mx,My,bx0,by0)
print('Listo SVD')
# ref_hole_FC=fouriercont2D_comp(def_hole,Mx,My,bx0,by0,D,U,S,V)
# print('Listo Refo')
def_hole_FC=fouriercont2D_comp(def_hole,Mx,My,bx0,by0,D,U,S,V)
print('Listo Defo')
# toc()

# print(np.shape(def_hole_FC))

plt.figure()
plt.imshow(def_hole_FC[:,:175])
plt.colorbar()

plt.figure()
plt.imshow(def_hole[:,:175])
plt.colorbar()

print(np.shape(def_hole),np.shape(def_hole_FC))
#%%
