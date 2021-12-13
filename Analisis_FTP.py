import numpy as np
import matplotlib.pyplot as plt
import Func_Splines as spl
import Continuacion_Fourier as cf
from skimage.io import imread
from scipy.interpolate import splev
import h5py 
plt.ion()
#%%
# traigo imagenes
gris = np.zeros((1024,1024))
for i in range(1,16):
    ni = '{:04d}'.format(i)
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0003\ID_0_C1S000300'+ni+'.tif')
#    ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0006/ID_0_C1S000600'+ni+'.tif')
    ima = np.array(ima,dtype='float')
    gris += ima
gris = gris/15

ref = np.zeros((1024,1024))
for i in range(1,14):
    ni = '{:04d}'.format(i)
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0002\ID_0_C1S000300'+ni+'.tif')
#    ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
    ima = np.array(ima,dtype='float')
    ref += ima
ref = ref/13
#%%
plt.figure()
plt.imshow(gris,cmap='gray')
plt.show()
plt.figure()
plt.imshow(ref,cmap='gray')
plt.show()
plt.figure()
plt.imshow(ref-gris,cmap='gray')
plt.show()
#%%
num = 1
ni = '{:04d}'.format(num)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float')
del ima

plt.figure()
plt.imshow(im,cmap='gray')
plt.show()
plt.figure()
plt.imshow(im-gris,cmap='gray')
plt.show()
#%%
from scipy.signal import find_peaks
asd = ref - gris
xp = np.arange(0,1024,1)
x = 0.02086 * xp
n = 200
lin = asd[n,:] - np.mean(asd[n,:])

a, _ = find_peaks(lin,distance=10) #hay 82 picos, o sea, 81 longitudes de onda
d = (1014-9) / 81 * 0.02086
w = 2*np.pi/d
rec = np.cos(w*x + 1.5) * 75

plt.figure()
plt.plot(x,lin,'b-')
#plt.plot(a,lin[a],'ro')
plt.plot(x,rec,'r-')
plt.grid()
plt.show()
#%%
w = 2*np.pi/d
L, D = 79.6, 20.3
#%%
thx,thy, ns = 0.25, 45, 0.75 #0.5, 80, 0.5
t1 = time()
dphs = np.zeros((100,1024,1024))
for num in range(1,101):
    if num%10 == 0: print(num, end=' ')
    ni = '{:04d}'.format(num)
    ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
    #ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
    im = np.array(ima,dtype='float')
    dph, ft, gf = cf.dphase_2d(im,ref-gris,thx,thy,ns,inde=9)
    dphs[num-1] = dph
t2 = time()
print(t2-t1)
#%%
alt = (L*dphs) / (dphs - w*D)
#%%
plt.figure()
plt.imshow(alt[10])
plt.colorbar()
plt.show()
#%%
from mpl_toolkits.mplot3d import Axes3D
x = np.arange(0,1024,1)
xx, yy = np.meshgrid(x,x)

i,li,n = 2, 10, 10
at, xt, yt = alt[n,li:-li:i,li:-li:i], xx[li:-li:i,li:-li:i], yy[li:-li:i,li:-li:i] 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(xt, yt, at, color='0.75', rstride=10, cstride=10,cmap='magma')
plt.show()
#%%
from matplotlib import animation
i,li = 3, 10
at, xt, yt = alt[:,li:-li:i,li:-li:i], xx[li:-li:i,li:-li:i], yy[li:-li:i,li:-li:i] 

def update_plot(frame_number, at, plot):
    plot[0].remove()
    plot[0] = ax.plot_wireframe(xt, yt, at[frame_number], cmap="magma")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

fps = 20 # frame per sec
frn = 100 # frame number of the animation

plot = [ax.plot_wireframe(xt, yt, at[0], color='0.75', rstride=10, cstride=10)]
ax.set_zlim(2.5,-2.5)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(at, plot), interval=1000/fps)
#%%
means = []
for n in range(100):
    means.append(np.mean(alt[n]))

print(np.max(means), np.min(means))
#%%
#==============================================================================
# 
#==============================================================================
num = 800
ni = '{:04d}'.format(num)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima,dtype='float')

plt.figure()
plt.imshow(im)
plt.show()
#%%
thx,thy, ns = 0.5, 80, 0.75 #0.25, 45, 0.75
dph, ft, gf = cf.dphase_2d(im,ref-gris,thx,thy,ns,inde=9)
#%%
d = (1014-9) / 81 * 0.02086
w, L, D = 2*np.pi/d, 79.6, 20.3
alt = (L*dph[10:-10,10:-10]) / (dph[10:-10,10:-10] - w*D)

plt.figure()
plt.imshow(alt)
plt.colorbar()
plt.show()
#%%
from skimage.filters import gabor
rega = gabor(ref-gris,0.1)
imga = gabor(im,0.1)

thx,thy, ns = 0.25, 45, 0.75
dphga, ft, gf = cf.dphase_2d(imga[0],rega[0],thx,thy,ns,inde=9)

altga = (L*dphga[10:-10,10:-10]) / (dphga[10:-10,10:-10] - w*D)

plt.figure()
plt.imshow(imga[0])
plt.show()
#%%
#==============================================================================
# 
#==============================================================================
from skimage.restoration import inpaint
num = 1800
ni = '{:04d}'.format(num)
ima = imread(r'C:\Users\tomfe\Documents\TOMAS\Facultad\Laboratorio 6\Datos Buenos\ID_0_C1S0001\ID_0_C1S000100'+ni+'.tif')
#ima = imread(r'/home/paula/Documents/Fisica2021/L6y7/PFWT/ID_0_C1S0003/ID_0_C1S000300'+ni+'.tif')
im = np.array(ima[210:440,570:810],dtype='float')

mask = im>630
def_im = im * (1-mask)

t1 = time()
image_result = inpaint.inpaint_biharmonic(def_im, 1-mask)
t2 = time()
print(t2-t1)

plt.figure()
plt.imshow(def_im)
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(im)
plt.colorbar()
plt.show()
plt.figure()
plt.imshow(mask)
plt.show()
plt.figure()
plt.imshow(image_result)
plt.colorbar()
plt.show()

#%%
