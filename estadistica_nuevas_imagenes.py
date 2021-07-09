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

#Genero la fibra inicial, la final y la dinamica entre ellas
fib_i = gf.generar_fibra(N=100, L=1, alpha=0.1, Nt=8)
fib_f = gf.generar_fibra(N=100, L=1, alpha=0.1, Nt=8)
dinam = gf.generar_dinamica_entre_dos_instancias(fib_i, fib_f, Nsteps=10)

#Genero las imágenes con ruido de estas fibras
imagenes = [gf.genera_im_fibra(dinam[:, i]) for i in range(np.shape(dinam)[-1])]

#Veo algunas imágenes, para ver cómo son
for im in imagenes[4:7]:
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.colorbar()
    plt.show()
