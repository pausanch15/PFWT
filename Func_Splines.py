#Es lo que dice TODO en el Colab de Splines
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import thin, skeletonize, remove_small_objects
from scipy.interpolate import CubicSpline, splev, splrep, splprep
from scipy.signal import convolve2d, savgol_filter
from itertools import permutations

def encuentra_fibra(imagenes, connec=4):
    fibras = []
    for im in imagenes:
        im = np.asarray(im)
        binariza = np.mean(im.flatten())-3.5*np.std(im.flatten())
        im = im<binariza 
#        li = label(im)
        fibra = thin(remove_small_objects(im, connectivity=connec))
        fibras.append(fibra)
    return fibras

def cortar_fibra(fibra, cortar_ruido=True): # la fibra con el thin ya hecho
    kernel = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]])
    cf = convolve2d(fibra,kernel) # hago la convolución
    convolved_fibra = cf[1:-1,1:-1] * fibra # multiplico por la fibra para que quede 'thin'
    
    bordes = np.array(np.where(convolved_fibra == 2)) #saco los bordes como los puntos que sean 2
#  para dectetar si el borde es ruido lo que hago es ver si es vecino de un nudo (eso me lo da hacer 
# la convolucion de la convolucion). Si un borde real tambien esta cerca de un nudo lo pierdo 
# (si pierdo los dos bordes por lo anterior puede dar problemas)
    
    if cortar_ruido == True:
        cf2 = convolve2d(convolved_fibra,kernel)
        convolved2_fibra = cf2[1:-1,1:-1] * fibra
        bordes = np.array(np.where(convolved2_fibra == 5))
        no_bordes = np.array(np.where((convolved2_fibra == 6) | (convolved2_fibra == 7)))
        for i in range(len(no_bordes[0])): 
            a = list(no_bordes[:,i])
            convolved_fibra[a[0],a[1]] = 0 #saco este no borde
        cf3 = convolve2d(convolved2_fibra,kernel)
        convolved3_fibra = cf3[1:-1,1:-1] * fibra
        bordes = np.array(np.where(convolved3_fibra == 13))
        no_bordes = np.array(np.where(convolved3_fibra == 14))
        for i in range(len(no_bordes[0])):
            a = list(no_bordes[:,i])
            convolved_fibra[a[0],a[1]] = 0 #saco este no borde
    b = []
    for i in range(len(bordes[0,:])):
        bor = list(bordes[:,i])
        bor.reverse()
        b.append(bor)
    b = np.array(b) #son los bordes en lista [[b1x,b1y],[b2x,b2y]]
    
    nudos = np.array(np.where(convolved_fibra >= 4)) # encuentro los nudos con los puntos >= 4
    for i in range(len(nudos[0])):
        a = list(nudos[:,i])
        convolved_fibra[a[0],a[1]] = 0  #saco los nudos de la fibra para que quede partida
    
    bin_con_fibra = convolved_fibra > 0 #vuelvo a convertir la imagen en binaria
    secciones, cantidad_secciones = label(bin_con_fibra, return_num=True) #saco las secciones
    im_tramos = []
    for i in range(1,cantidad_secciones+1):
        secc = (secciones == i)
        im_tramos.append(secc) #obtengo los tramos
        
    tramos = []
    for im in im_tramos:
        coor = np.where(im > 0)
        y = coor[0]
        x = coor[1]
        tramo = []
        for i in range(len(y)):
            tramo.append([x[i],y[i]])
        tramos.append(np.array(tramo))

    return tramos,b


def buscar_bordes(x,y): #busca los bordes y nodos de la fibra  
    xb,yb, ind = [],[],[]
    for i in range(len(x)):
        xi,yi = x[i], y[i]
        vec = 0
        for j in range(len(x)):
            xj,yj = x[j], y[j]
            if j == i:
                continue
            if (np.abs(xi-xj))**2 + (np.abs(yi-yj))**2 <= 2:
                vec += 1
        if vec < 2:
            xb.append(xi)
            yb.append(yi)
            ind.append(i)
    return xb,yb,ind

def primer_paso(xt,yt,xd,yd): #hago el primer paso
    for i in range(len(xd)):
        if (np.abs(xt[-1]-xd[i]))**2 + (np.abs(yt[-1]-yd[i]))**2 <= 2:
            xt.append(xd[i])
            yt.append(yd[i])
            del xd[i]
            del yd[i]
            break

def a_borde(xt,yt,xd,yd,xb,yb): #avanzo por la fibra hasta llegar a un nodo o un borde
    bb = [ [xb[i],yb[i]] for i in range(len(xb)) ]
    while [xt[-1],yt[-1]] not in bb: #  ((xt[-1] not in xb) or (yt[-1]  yb[1])): 
        for i in range(len(xd)):
            if (np.abs(xt[-1]-xd[i]))**2 + (np.abs(yt[-1]-yd[i]))**2 <= 2: 
                xt.append(xd[i])
                yt.append(yd[i])
                del xd[i]
                del yd[i]
                break
                
def ordenar_tramo(tramo):
    x,y = tramo[:,0], tramo[:,1]
    xb,yb,ind = buscar_bordes(x,y)
    xd, yd = list(x),list(y)
    xt, yt = [xb[0]],[yb[0]]
    del xd[ind[0]]
    del yd[ind[0]]
    primer_paso(xt,yt,xd,yd)
    a_borde(xt,yt,xd,yd,xb,yb)
    tramo_ord = []
    for i in range(len(xt)):
        tramo_ord.append([xt[i],yt[i]])
    return np.array(tramo_ord)

def ordenar_fibra(tramos):
    fibra = []
    for i in range(len(tramos)):
        tramo = tramos[i]
        tr_ord = ordenar_tramo(tramo)
        fibra.append(tr_ord)
    return fibra


def curvatura(fib,window=21,s=10):
    x,y = fib[:,0],fib[:,1]
    porder = 3
    xsg = savgol_filter(x,window,porder)
    ysg = savgol_filter(y,window,porder)
    spl, u = splprep([xsg,ysg],s=s)
    t_spl = np.linspace(0,1,100000)
    spline = splev(t_spl,spl)
    spline_1d = splev(t_spl,spl,der=1)
    spline_2d = splev(t_spl,spl,der=2)
    curv = np.abs(spline_2d[0] * spline_1d[1] - spline_1d[0] * spline_2d[1]) / ((spline_1d[0])**2 + (spline_1d[1])**2)**(3/2)
    return curv, np.mean(curv), np.std(curv), spl # t_spl, spline

def pegar_fibra(tramos,bordes,tamano_nudo=30, window=21, s=10):
    if len(tramos) == 1:
        cur,m_curv,_,spl = curvatura(tramos[0],window=window,s=s)
        return cur, spl#t_spl, spline[0],spline[1]
    
    if len(bordes) == 1:
        tra_medios = []
        for i in range(len(tramos)):
            if list(bordes[0]) in tramos[i].tolist():
                pri_tramo = tramos[i]
            else:
                tra_medios.append(tramos[i])
        if not list(bordes[0]) == list(pri_tramo[0]):
            pri_tramo = np.flip(pri_tramo,axis=0)
        
        nudo = []
        for i in range(len(tramos)):
            if len(tramos[i]) < tamano_nudo:
                nudo.append(i)
        nudo.reverse()
        for i in nudo:
            del tramos[i]

        pos_tra_med = []
        for i in range(len(tra_medios)):
            pos_tra_med.append( ( tra_medios[i],np.flip(tra_medios[i],axis=0) ))
        n = len(pos_tra_med)
        perm = list(permutations(range(n),n))
        perm_inv = list(set(list(permutations([0,1]*n, n))))
        min_curv = 10**3
        for i in range(len(perm)):
            for k in range(len(perm_inv)):
                orden = []
                for j,l in zip(perm[i],perm_inv[k]):
                    orden.append(pos_tra_med[j][l])
                orden.insert(0,pri_tramo)
                fib = np.concatenate(tuple(orden))
                cur,m_curv,_,spl = curvatura(fib,window=window,s=s)
                if m_curv < min_curv:
                    min_curv = m_curv
                    curvatur = cur
                    spli = spl
#                    t,xf,yf = t_spl,spline[0],spline[1]

        return curvatur, spli #t,xf,yf


    tra_medios = []
    for i in range(len(tramos)):
        if list(bordes[0]) in tramos[i].tolist():
            pri_tramo = tramos[i]
        elif list(bordes[1]) in tramos[i].tolist():
            ult_tramo = tramos[i]
        else:
            tra_medios.append(tramos[i])

    if not list(bordes[0]) == list(pri_tramo[0]):
        pri_tramo = np.flip(pri_tramo,axis=0)
    if not list(bordes[1]) == list(ult_tramo[-1]):
        ult_tramo = np.flip(ult_tramo,axis=0)    
    
    nudo = []
    for i in range(len(tramos)):
        if len(tramos[i]) < tamano_nudo and bordes[1] not in tramos[i] and bordes[0] not in tramos[i]:
            nudo.append(i)
    nudo.reverse()
    for i in nudo:
        del tramos[i]

    pos_tra_med = []
    for i in range(len(tra_medios)):
        pos_tra_med.append( ( tra_medios[i],np.flip(tra_medios[i],axis=0) ))

    n = len(pos_tra_med)
    perm = list(permutations(range(n),n))
    perm_inv = list(set(list(permutations([0,1]*n, n))))
    min_curv = 10**3
    for i in range(len(perm)):
        for k in range(len(perm_inv)):
            orden = []
            for j,l in zip(perm[i],perm_inv[k]):
                orden.append(pos_tra_med[j][l])
            orden.insert(0,pri_tramo)
            orden.append(ult_tramo)
            fib = np.concatenate(tuple(orden))
            cur,m_curv,_,spl = curvatura(fib,window=window,s=s)
            if m_curv < min_curv:
                min_curv = m_curv
                curvatur = cur
                spli = spl
#                t,xf,yf = t_spl,spline[0],spline[1]

    return curvatur, spli #t,xf,yf

