# demo ftp completo

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.ion()
from matplotlib import cm
import numpy as np

# Lx y Ly eran 315
Lx, Ly = 150, 150 
lambdaa=8

x,y=np.meshgrid(np.arange(Lx),np.arange(Ly))

Z=np.random.normal(1200,200,25)
B=Lx*(np.random.rand(25)*0.9+0.05)
C=Ly*(np.random.rand(25)*0.9+0.05)
signo=np.sign(2*np.random.rand(25)-1)
iphi=np.zeros(np.shape(x))

for i in range (0,25):
    iphi=iphi+signo[i]*np.exp((-(x-C[i])**2-(y-B[i])**2)/Z[i])

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(x, y, iphi, cmap=cm.coolwarm,linewidth=0, antialiased=False)
# plt.title('Fase Original')
 # plt.show()

R=0.*np.random.rand(1)/10

refo=np.cos(2*np.pi/lambdaa*x) + R*(-1+2*np.random.rand(Lx,Ly))
defo=np.cos(2*np.pi/lambdaa*x+iphi) + R*(-1+2*np.random.rand(Lx,Ly))

# f, (ax1, ax2) = plt.subplots(1, 2)
# ax1.imshow(refo)
# ax1.set_title('Referencia')
# ax2.imshow(defo)
# ax2.set_title('Deformada')
# plt.show()

from imagetodephasemap1D import imagestodphasemap1D

s = 0.9 # valor original
ns = 0.2 #valor original
rphi = imagestodphasemap1D(defo,refo,s,ns)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(x, y, rphi, cmap=cm.coolwarm,linewidth=0, antialiased=False)
# plt.title('Fase Recuperada sin agujero')
# plt.show()

porcion=[0,315];

maskoutholeaux=np.ones(np.shape(refo[porcion[0]:porcion[1],porcion[0]:porcion[1]]));


# era a=40, rad=8
# rad=10.
# a=200.
# aux=0.
# xx=np.arange(porcion[0]+2.5*rad,porcion[1]-1.5*rad,a)
# yy=np.arange(porcion[0]+2.5*rad,porcion[1]-1.5*rad,a)

# for ll in range (0,len(yy)):
#     for kk in range (0,len(xx)):
#         maskoutholeaux[np.sqrt((x[porcion[0]:porcion[1],porcion[0]:porcion[1]]-xx[kk])**2+(y[porcion[0]:porcion[1],porcion[0]:porcion[1]]-yy[ll])**2)<rad]=0
# maskouthole=np.ones(np.shape(refo))
# maskouthole[porcion[0]:porcion[1],porcion[0]:porcion[1]]=maskoutholeaux

# RNAN=np.sum(maskouthole==0)/((porcion[1]-porcion[0])**2-np.sum(maskouthole==0))

# un solo agujero centrado
radx = 3 
rady = 40
# maskouthole=np.ones(np.shape(refo))
maskouthole = ((x-int(Lx/2))**2/radx**2 + (y-int(Ly/2))**2/rady**2 >= 1)

# Aplicamos las mascaras

ref0 = refo*maskouthole;
def0 = defo*maskouthole;

plt.close('all')
plt.figure()
plt.imshow(def0)

rphiWH = imagestodphasemap1D(def0,ref0,s,ns);

rphiWH[maskouthole==0] = np.nan;

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(x, y, rphi-iphi, cmap=cm.coolwarm,linewidth=0, antialiased=False)
# plt.title('Diferencia entre Fase Recuperada y Original')
# plt.show()

#  Fourier Continuation of one of the holes.

d1,d2=np.shape(refo);

ref_hole=refo.copy()
def_hole=defo.copy()

ref_hole[maskouthole== 0]=np.nan
def_hole[maskouthole== 0]=np.nan

# Determinacion de Mx y My

k=np.arange(0,(Lx)/2)/(Lx)
kxs=np.zeros(Lx)

for ii in range (0,Lx):
    fftdefx=np.fft.fft(def_hole[ii,:])
    P2=np.abs(fftdefx/Lx)
    P1=P2[0:int(np.floor(Lx/2+1))]
    P1[1:-1]=2*P1[1:-1]
    imax=np.argmax(np.abs(P1))
    kxs[ii]=k[imax]

kx=np.sum(kxs[kxs!=0])/len(kxs[kxs!=0])
bx0=(.8/kx+d2)/d1
by0=(.8/kx+d2)/d1
Mx = int(np.floor(bx0*Lx*2*kx))

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
# from tictoc import tic, toc
from generarD import generarD
from fouriercont2D_comp import fouriercont2D_comp

# tic()
D, U ,S, V=generarD(ref_hole,Mx,My,bx0,by0)
print('Listo SVD')
ref_hole_FC=fouriercont2D_comp(ref_hole,Mx,My,bx0,by0,D,U,S,V)
print('Listo Refo')
def_hole_FC=fouriercont2D_comp(def_hole,Mx,My,bx0,by0,D,U,S,V)
print('Listo Defo')
# toc()

print(np.shape(def_hole_FC))

aux1=ref_hole_FC.copy()
aux2=def_hole_FC.copy()
aux1[0:d1,0:d2]=ref_hole.copy()
aux2[0:d1,0:d2]=def_hole.copy()

ref_nohole=aux1.copy()
ref_nohole[np.isnan(aux1)]=ref_hole_FC[np.isnan(aux1)]
def_nohole=aux2.copy()
def_nohole[np.isnan(aux2)]=def_hole_FC[np.isnan(aux2)]

f, ((ax1, ax2),(ax3,ax4))= plt.subplots(2, 2)
ax1.imshow(ref_hole)
ax1.set_title('Referencia agujereada')
ax2.imshow(def_hole)
ax2.set_title('Deformada agujereada')
ax3.imshow(ref_nohole)
ax3.set_title('Referencia rellena')
ax4.imshow(def_nohole)
ax4.set_title('Deformada rellena')
# plt.show()

# stopstop


#maskouthole3=np.ones(np.shape(def_nohole));
#maskouthole3[0:d1,0:d2]=maskouthole;

#rphiWOH2 = imagestodphasemap1D(def_nohole,ref_nohole,s,ns);
#rphiWOH2 [maskouthole3==0] = np.nan

#rphiWOH=rphiWOH2[0:d1,0:d2]
#rphiWOH[maskouthole==0] = np.nan

#hh = np.max(np.abs(rphiWOH[~np.isnan(rphiWOH)]-iphi[~np.isnan(rphiWOH)]))/np.max(np.abs(rphi-iphi))
#print(hh)

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(x, y, np.abs(rphiWOH-rphi), cmap=cm.coolwarm,linewidth=0, antialiased=False)
#plt.title('relllenada y recuperada original')
#plt.show()

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(x, y, np.abs(rphiWH-rphi), cmap=cm.coolwarm,linewidth=0, antialiased=False)
#plt.title('agujereada y recuperada original')
#plt.show()

#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(x, y, np.abs(rphiWOH-iphi), cmap=cm.coolwarm,linewidth=0, antialiased=False)
#plt.title('rellenada y original')
## plt.show()
##
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(x, y, np.abs(rphi-iphi), cmap=cm.coolwarm,linewidth=0, antialiased=False)
#plt.title('recuperada original y original')
## plt.show()

#fig = plt.figure(figsize=plt.figaspect(0.5))
## set up the axes for the first plot
#ax1 = fig.add_subplot(1, 2, 1, projection='3d')
#ax1.plot_wireframe(x, y, np.abs(rphiWOH-iphi), rstride=10, cstride=10)
#ax1.set_title('Diferencia entre original y rellenada')
#ax1.set_zlim(0., 0.8)
#ax = fig.add_subplot(1, 2, 2, projection='3d', sharex=ax1, sharey=ax1,
#        sharez=ax1)
#ax.plot_wireframe(x, y, np.abs(rphi-iphi), rstride=10, cstride=10)
#ax.set_title('Diferencia entre original y recuperada sin agujeros')
#ax.set_zlim(0., 0.8)
## plt.show()
