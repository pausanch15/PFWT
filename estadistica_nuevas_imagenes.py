#Acá pruebo hacer estadística con las imgágenes generadas con los códigos que pasó Pablo y las funciones mejoradas.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from scipy.interpolate import splev, splrep, splprep
from skimage.morphology import thin, skeletonize, remove_small_objects, binary_dilation, dilation
from skimage.util import random_noise
from skimage.filters import gaussian
import Func_Genera_Fibras_2 as gf
plt.ion()
import Func_Splines as spl

#Genero la fibra inicial, la final y la dinamica entre ellas
fib_i = gf.generar_fibra(N=100, L=1, alpha=0.1, Nt=8)
fib_f = gf.generar_fibra(N=100, L=1, alpha=0.1, Nt=8)
dinam = gf.generar_dinamica_entre_dos_instancias(fib_i, fib_f, Nsteps=10)

#Genero las imágenes con ruido de estas fibras
imagenes = [gf.genera_im_fibra(dinam[:, i]) for i in range(np.shape(dinam)[-1])]

#Veo algunas imágenes, para ver cómo son
# for im in imagenes[4:7]:
    # plt.figure()
    # plt.imshow(im, cmap='gray')
    # plt.colorbar()
    # plt.show()

#Trato de encontrar las fibras en cada imagen
fibras = spl.encuentra_fibra(imagenes)

#Veo cómo encuentra las fibras
# for im in fibras[4:7]:
    # plt.figure()
    # plt.imshow(im, cmap='gray')
    # plt.colorbar()
    # plt.show()

#Hago lo mismo que hacíamos antes para hacer estadística
#Elijo una de las fibras que encontramos en las imágenes, cualquiera.
ff = 4
fibra = fibras[ff]

#Interpolo la fibra encontrada
tramos, bordes = spl.cortar_fibra(fibra, cortar_ruido=True)
tramos = spl.ordenar_fibra(tramos)
curv, spline = spl.pegar_fibra(tramos, bordes, window=21, s=10)

t_spl = np.linspace(0, 1, 10000)

#xf y yf son as coordenadas de la fibra
xf, yf = splev(t_spl, spline)

#Vemos la fibra que recuperamos
plt.figure()
plt.imshow(fibra)    
plt.plot(xf, yf, 'r-')
plt.show()

#xo y yo son las coordenadas de la fibra original. Es la fibra que está en dinam[:, ff]
y, x = np.real(dinam[:, ff]), np.imag(dinam[:, ff])
spl, u = splprep([x, y], s=0)
xo, yo = splev(t_spl, spl)

#Comparo la fibra original con la que analizamos
plt.figure()
plt.plot(xo, yo, 'g-', label='Fibra Original') 
plt.plot(xf, yf, 'r-', label='Fibra Obtenida')
plt.legend()
plt.show()

plt.figure()
plt.plot(xf-xo, label='Diferencia en x')
plt.plot(yf-yo, label='Diferencia en y')
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.plot(np.abs(xf-xo), label='Diferencia en x')
plt.plot(np.abs(yf-yo), label='Diferencia en y')
plt.grid()
plt.legend()
plt.show()

#Hago el histograma para la diferencia en x y en y. Cuantos puntos difieren en cada valor?
plt.figure()
plt.hist(xf-xo, bins=30, facecolor='blue', label='x', alpha=0.5, edgecolor='k')
plt.hist(yf-yo, bins=30, facecolor='red', label='y', alpha=0.5, edgecolor='k')
plt.legend()
plt.show()

plt.figure()
plt.hist(np.abs(xf-xo), bins=30, facecolor='blue', label='x', alpha=0.5, edgecolor='k')
plt.hist(np.abs(yf-yo), bins=30, facecolor='red', label='y', alpha=0.5, edgecolor='k')
plt.legend()
plt.show()
