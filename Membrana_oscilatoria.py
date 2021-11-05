import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
#%%
x = np.linspace(0,1,200)
y = np.linspace(0,1,200)
t = np.linspace(0,10,100)

xx, yy = np.meshgrid(x,y)
c = 0.5
mmax,nmax = 10,10
h = np.zeros((100,200,200))
for i in range(len(t)):
    for m in range(mmax):
        for n in range(nmax):
            w = c*np.pi * np.sqrt(m**2+n**2)
            A,B = 1/mmax,1/nmax
            h[i] += np.sin(m*np.pi*xx) * np.sin(n*np.pi*yy) * (A * np.cos(w*t[i]) + B * np.sin(w*t[i]))
#%%
fps = 10 # frame per sec
frn = 100 # frame number of the animation

def update_plot(frame_number, h, plot):
    plot[0].remove()
    plot[0] = ax.plot_wireframe(xx, yy, h[frame_number], cmap="magma")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot = [ax.plot_wireframe(xx, yy, h[0], color='0.75', rstride=1, cstride=1)]
ax.set_zlim(2.5,-2.5)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(h, plot), interval=1000/fps)
#%%
from time import time
t1 = time()
hft = np.fft.fftn(h)
w, kx, ky = np.fft.fftfreq(100), np.fft.fftfreq(200), np.fft.fftfreq(200) 
t2 = time()
t2-t1
#%%
kxx,kyy = np.meshgrid(kx,ky)
n = 99
fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_wireframe(kxx,kyy,np.abs(hft[n,:,:]))
#ax.colorbar()
plt.imshow(np.log(np.abs(np.fft.fftshift(hft[:,:,n]))))
plt.colorbar()
plt.show()
#%%
plt.figure()
plt.plot(w,np.abs(hft[:,20,20]))
plt.show()


