import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from scipy.interpolate import splev, splrep, splprep
from skimage.morphology import thin, skeletonize, remove_small_objects, binary_dilation, dilation
from skimage.draw import line
from skimage.util import random_noise
from skimage.filters import gaussian

def largo(x):
    """
    Calcula el largo de una curva (muy aproximado)
    """
    x = np.diff(x)
    return np.sum(np.abs(x))

def establecer_largo(x, largo_objetivo=1):
    """
    Toma una curva en 2D y la lleva al largo objetivo (default = 1).
    """
    largo_actual = largo(x)
    xout = x*(largo_objetivo/largo_actual)
    return xout

def generar_fibra(N=100, L=1, alpha=0.1, Nt=8):
    """
    Genera una fibra deformada suavemente de numero de puntos y largo dado
    N = nro puntos que la integran
    L = largo (adimensional) 
    alpha = cuan compleja es (alpha de preferencia entre 0 y 1)
    """

    t = np.linspace(0, alpha*2*np.pi, N)
    r = np.zeros_like(t)
    for k in range(Nt):
        ampsin = np.random.randn(Nt)
        ampcos = np.random.randn(Nt)
        r += ampcos[k]*np.cos(k*t) 
        r += ampsin[k]*np.sin(k*t) 

    x = r*np.cos(t) + 1j*r*np.sin(t)
    x = establecer_largo(x, L)
    x = x - x[0]
    return x

def generar_dinamica_entre_dos_instancias(xi, xf, Nsteps=10):
    """
    Genera la dinamica intermedia entre dos estados conformacionales de la fibra
    """
    xyout = np.zeros((len(xi), Nsteps)).astype(complex)

    k = 0
    for t in np.linspace(0, 1, Nsteps): 
        xint = (xf-xi)*t + xi

        # fijamos el largo para todo instante posterior (default a 1)
        if t > 0:
            xint = establecer_largo(xint)
       
        xyout[:,k] = xint 
        k += 1
    return xyout

def generar_dinamica(xlist, Nsteps=10):
    """
    Genera una dinamica compleja para la fibra
    vinculando varios estadios intermedios
    """
    # xlist es lista de arrays complejos    
    # xlist = (x1, x2, x3, ...)
    N_puntos = len(xlist[0])
    N_transiciones = len(xlist)-1

    xyout = np.zeros((N_puntos, (N_transiciones)*Nsteps)).astype(complex)

    for k in range(N_transiciones):
        xi = xlist[k]     
        xf = xlist[k+1]
        xytemp = generar_dinamica_entre_dos_instancias(xi, xf, Nsteps=Nsteps)
        xyout[:,k*Nsteps:(k+1)*Nsteps] = xytemp
    return xyout

def genera_im_fibra(fibra, bins=500, sigma=1, ruido=0.003, fondo=0.05):
    """
    Genera las imágenes de las fibras, con los ruidos de siempre.
    fibra = fibra generada con generar_fibra
    """
    x_f, y_f = np.real(fibra), np.imag(fibra)
    spl, u = splprep([x_f, y_f], s=0)
    t_spl = np.linspace(0, 1, len(x_f)*100)
    fr = splev(t_spl, spl)
    
    im_fib, x_ed, y_ed = np.histogram2d(*fr, bins) #Imagen con la fibra
    
    im_fibra = np.zeros((1000, 1000)) #Imagen más grande para poner la fibra
    im_fibra[100:600, 100:600] = im_fib #Agrego la fibra
    
    im_fibra = im_fibra==0
    im_fibra = dilation(1-im_fibra)
    im_fibra = np.array(1-im_fibra,dtype='float')
    im_fibra = random_noise(im_fibra, mode='s&p', amount=ruido)
    # ff = np.linspace(0,999,1000)
    # fx,fy = np.meshgrid(ff,ff)
    # im_fibra = (im_fibra + fx*fy / 1000**2 * fondo) / 2
    im_fibra = gaussian(im_fibra, sigma=sigma)
    im_fibra = np.array(255*im_fibra, dtype='uint8')
    
    return im_fibra

def fibra_a_skeleton(xfibra, L=300, S=(1024, 1024), graph=False):
    """
    usamos Bresenham’s line algorithm para la conversion,
    previo escalado de la fibra original
    """
    x, y = np.real(xfibra), np.imag(xfibra)
    # hacemos que sea positiva toda la senal
    x -= np.min(x)
    y -= np.min(y)
    # la escalamos
    x *= L
    y *= L
    # la posicionamos cerca del centro
    x += S[1]//2
    y += S[0]//2
    # imagen fondo blanco (high) fibra negra (low)
    im = np.ones(S, dtype=np.uint8)
    for i in range(len(x)-1):
        rr, cc = line(int(x[i]), int(y[i]), int(x[i+1]), int(y[i+1]))
        im[rr, cc] = 0
    if graph == True:
        plt.figure()
        plt.imshow(im, cmap='gray')
        plt.plot(y, x, 'r-o', linewidth=0.75)
        plt.xlim(0, S[1])
        plt.ylim(0, S[0])
        plt.colorbar()
    return im, x,y 

def curva(x,y,s=0):
#    porder = 3
    spl, u = splprep([x,y],s=s)
    t_spl = np.linspace(0,1,100000)
#    spline = splev(t_spl,spl)
    spline_1d = splev(t_spl,spl,der=1)
    spline_2d = splev(t_spl,spl,der=2)
    curv = np.abs(spline_2d[0] * spline_1d[1] - spline_1d[0] * spline_2d[1]) / ((spline_1d[0])**2 + (spline_1d[1])**2)**(3/2)
    return np.max(curv) # t_spl, spline

def genera_im_dinamica(frames=1, n_fibras=2, L=300, sigma=1, dilat=True,
                       ruido=0.003, fondo=0.05, alpha=0.1, Nt=8, N=100, curvatura=1.5):
    """
    Genera las imagenes de la dinamica de fibras. Esta función ya genera las fibras.
    n_fibras = la cantidad de fibras que se usa para hacer la dinamica, tiene que ser mayor o igual a 2
    frames = cantidad de frames entre fibras (el Nsteps de generar_dinamica)
    """
    fibras = []
    for i in range(n_fibras):
        cur_m = curvatura+2
#        print(cur_m)
        while cur_m > curvatura:
            fib = generar_fibra(N=N, L=1, alpha=alpha, Nt=Nt)
            x,y = np.real(fib), np.imag(fib)
            cur_m = curva(x,y)
#            print(cur_m)
#        
        fibras.append(fib)
        
    dinam = generar_dinamica(fibras,frames)
   
    imagenes = []
    splines = []
    for i in range( (n_fibras-1)*frames ):
        
        ima,x_f,y_f = fibra_a_skeleton(dinam[:,i], L=L)
            
        spl, u = splprep([x_f, y_f], s=0)
        splines.append(spl)
        
#        im_fibra, x_ed, y_ed = np.histogram2d(*fr, 1000, [[0,1000],[0,1000]])
    
#        dr = drift*np.random.random(2) - drift/2
#        ox += int(dr[0])
#        oy += int(dr[1])
#        print(dr,' ',ox,ox+bins,' ',oy,oy+bins)
        
#        im_fibra = np.zeros((1000, 1000)) #Imagen más grande para poner la fibra
#        im_fibra[ox:(ox+bins), oy:(oy+bins)] = im_fib
    
        im_fibra = ima==1
        if dilat == True:
            im_fibra = dilation(1-im_fibra)
            im_fibra = np.array(1-im_fibra,dtype='float')
        im_fibra = random_noise(im_fibra, mode='s&p', amount=ruido)
        ff = np.linspace(0,999,1024)
        fx,fy = np.meshgrid(ff,ff)
        im_fibra = (im_fibra + fx*fy / 1024**2 * fondo) / 2
        im_fibra = gaussian(im_fibra, sigma=sigma)
        im_fibra = np.array(255*im_fibra, dtype='uint8')
        imagenes.append(im_fibra)
        
    return imagenes, splines
        
