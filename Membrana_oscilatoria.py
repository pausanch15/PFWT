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
m,n = 2,2
h = np.zeros((100,200,200))
for i in range(len(t)):
    for m in [1,2,3]:
        for n in [1,2,3]:
            w = c*np.pi * np.sqrt(m**2+n**2)
            A,B = 1,1
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
ax.set_zlim(8.5,-8.5)
ani = animation.FuncAnimation(fig, update_plot, frn, fargs=(h, plot), interval=1000/fps)
    