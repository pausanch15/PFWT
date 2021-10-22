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
from skimage.filters import laplace, sobel, threshold_local
from skimage.morphology import thin, skeletonize, remove_small_objects, dilation, binary_erosion
from skimage.measure import label, regionprops
from scipy.interpolate import splev, splrep, splprep
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
num = 568 #568, 165
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
lin = 380
iml = im[380,:]
img = gris[380,:]
imlg = img[380]
plt.figure()
plt.plot(iml,'-o')
plt.plot(img)
plt.show()
plt.figure()
plt.imshow(im-0.8*gris)
plt.show()
#%%
def rec_fibra(imagenes,gris,connec=0,std_mul=2.2):
    fibras, bbs = [], []
    griss = np.asarray(gris)
    for im_n, im in enumerate(imagenes):
        im = np.asarray(im)
        imlap = sobel(laplace(im-0.8*griss,ksize=3),mode='wrap')
        binariza = np.mean(imlap)+std_mul*np.std(imlap)
        print(np.mean(imlap),np.std(imlap), binariza)
        imb = imlap>binariza
        bor = 5
        fibra = remove_small_objects(imb[bor:-bor,bor:-bor], connectivity=0)
#        li = label(fibra)
# hasta aca es nuevo, para cuando aparece más de un coso despues de binarizar (datos del labo)    
        prop = regionprops(fibra.astype(int))
        try:
            bb = prop[0].bbox
            bbs.append(np.array(bb)+bor)
            recorte = fibra[bb[0]:bb[2], bb[1]:bb[3]]
            fibra_t = binary_erosion(dilation(recorte))
#            fibra_t = binary_erosion(dilation(recorte))#recorte
#            fibra_t = skeletonize(recorte)
            # fibra2 = thin(fibra[1:500, 1:500])
#            fibra[bb[0]:bb[2], bb[1]:bb[3]] = fibra_t
    #        fibras.append(fibra)
            fibras.append(fibra_t)
        except:
            print(f'Falló en la imagen {im_n}')
#            print(prop)
#            print()
            pass
    return fibras, bbs
#%%
num = 166 #568, 165, 444, 569, 3070
ni = '{:04d}'.format(num)
im = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Labo\FTP_fibra\ID_0_C1S000500'+ni+'.tif')

mul = 2.46
fibra, bb = rec_fibra([im],gris,std_mul=mul)
#tramos,bordes = spl.cortar_fibra_rap(fibra[0],bb[0],cortar_ruido=True)
#tramos = spl.ordenar_fibra(tramos)
#try:
#    curv,spline = spl.pegar_fibra(tramos,bordes,window=33,s=10)
#except: print('Jajaja')

#t_spl = np.linspace(0, 1, 10000)
#xf, yf = splev(t_spl, spline)

#print(np.max(curv))
#
#plt.figure()
#plt.set_cmap('gray')
#plt.imshow(im-0.8*gris)
#plt.plot(xf, yf, 'r-')
#plt.show()

print(bb)
plt.figure()
plt.imshow(fibra[0])
plt.show()
#%%
bbs = bb[0]
imr = im[bbs[0]:bbs[2], bbs[1]:bbs[3]]
im_nofib = imr * (1-fibra[0]) 
im_nofib = im_nofib.astype(float)
im_nofib[im_nofib == 0] = np.nan
imn = im[:,:]
imn = imn.astype('float')
imn[bbs[0]:bbs[2], bbs[1]:bbs[3]] = im_nofib 

#refr = ref[bbs[0]:bbs[2], bbs[1]:bbs[3]]
#ref_nofib = refr * (1-fibra[0]) + np.mean(refr) * fibra[0]
#refn = ref[:,:]
#refn[bbs[0]:bbs[2], bbs[1]:bbs[3]] = ref_nofib 

#imn[imn==0] = np.nan
ref = ref.astype('float')

plt.figure()
plt.imshow(imn)
plt.colorbar()
plt.show()

#np.save('imn',imn)

plt.figure()
plt.imshow(ref)
#plt.colorbar()
plt.show()
#%%
#-------------------------------------------------------------------------------------------------
#%%
#FTP, donde esta la fibra se ve medio raro
#imn = np.load('imn.npy')
from generarD import generarD
from fouriercont2D_comp import fouriercont2D_comp


imnr = imn[300:450, 200:350]
def_hole = imnr
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
plt.imshow(def_hole_FC[:150,:150])
plt.colorbar()

plt.figure()
plt.imshow(def_hole[:150,:150])
plt.colorbar()

print(np.shape(def_hole),np.shape(def_hole_FC))
#%%
imn[300:450, 200:350] = def_hole_FC[:150,:150] 

plt.figure()
plt.imshow(imn-gris)
plt.show()
plt.figure()
plt.imshow(ref-gris)
plt.show()
plt.figure()
plt.imshow(gris)
plt.show()

#%%
thx,thy, ns = 0.5, 80, 0.5
gris = gris.astype('float')
dp, fY0, gf = dphase_2d(imn-gris, ref-gris ,thx,thy,ns,inde=9)

lim = 10
plt.figure()
plt.imshow(dp[lim:-lim,lim:-lim])
plt.title('thx={}, thy={}, ns={}'.format(thx,thy,ns))
plt.colorbar()
plt.show()
#%%
aa = imn-gris
fY0 = np.fft.fft2(aa-np.min(aa))

plt.figure()
plt.plot(np.abs(fY0[0,:]))
plt.show()
print(np.abs(fY0[0,:]))
#%%
plt.figure()
plt.imshow(aa)
plt.colorbar()
plt.show()
#%%
for i, num in enumerate(aa):
    if num == 'nan':
        print(i)
#%%
l = 26
c = 1003
print(aa[l,c])


#%%