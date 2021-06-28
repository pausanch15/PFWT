#Aca trato de interpolar la fibra de una forma alternativa
#Voy a adaptar el codigo que mando Pablo al Slack
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.morphology import thin, skeletonize, erosion, opening
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import scipy.ndimage as ndi
import networkx as nx
import Func_Splines as spl
from skimage.color import rgb2gray
from scipy.interpolate import CubicSpline

#Vuelvo a probar con aquella querida imagen de prueba
image = Image.open('BrouzetMaster.png')
im = np.asarray(image, dtype=float)[:, :, 0]
im = im[400:600, 250:850]

#Encuentro la fibra con los codigos nuevos
fibra = spl.encuentra_fibra([im])[0]

#Aca empieza lo nuevo
blobs = label(fibra, connectivity=1)
pb = regionprops(blobs)

#Aca vamos recorriendo la fibra, y calculando el centroide de cada grupo de puntos para quedarnos con el trazo de la fibra un poco menos detallado.
points = []
for parxy in pb:
    points.append([parxy.centroid[1], parxy.centroid[0]])
points = np.array(points)

#3 branch branching
selems = list()

selems.append(np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 0, 0]]))
selems.append(np.array([[1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0]]))
selems.append(np.array([[1, 0, 1],
                        [0, 1, 0],
                        [0, 1, 0]]))
selems.append(np.array([[0, 1, 0],
                        [1, 1, 0],
                        [0, 0, 1]]))
selems = [np.rot90(selems[i], k=j) for i in range(4) for j in range(4)]

#4 branch branching
selems.append(np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]]))
selems.append(np.array([[1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 1]]))

branches = np.zeros_like(fibra, dtype=bool)
for selem in selems:
    branches |= ndi.binary_hit_or_miss(fibra, selem)

#Endpoints
#El operador OR bit a bit (|) es un operador binario que toma dos patrones de bits de igual longitud y realiza la operación OR lógica en cada par de bits correspondientes. Devuelve 1 si uno o ambos bits en la misma posición son 1; de lo contrario, devuelve 0. 
selems = list()

selems.append(np.array([[0, 1, 0],
                        [0, 1, 0],
                        [0, 0, 0]]))
selems.append(np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]]))
selems = [np.rot90(selems[i], k=j) for i in range(2) for j in range(4)]

endpoints = np.zeros_like(fibra, dtype=bool)
for selem in selems:
    endpoints |= ndi.binary_hit_or_miss(fibra, selem)

#%%
#Graficamos estos pasos.
# plt.figure()
# plt.imshow(branches.astype(float) + fibra.astype(float))
# plt.title('braching points')
# plt.colorbar()
# plt.show()
# 
# plt.figure()
# plt.imshow(endpoints.astype(float) + fibra.astype(float))
# plt.title('endpoints')
# plt.colorbar()
# plt.show()
# 
# plt.figure()
# plt.imshow(np.logical_xor(branches, fibra))
# plt.title('logical xor entre branches y skfiber')
# plt.show()
#%%

#Separamos la fibra en ramas y hacemos lo de los centriodes en cada una.
fibra_split = np.logical_xor(branches, fibra)
lis = label(fibra_split)

#Para la rama recta
blobs = label(lis==2, connectivity=1)
pb = regionprops(blobs)
points = []
for parxy in pb:
     points.append([parxy.centroid[1], parxy.centroid[0]])
points = np.array(points)

#Para la rama del rulo
blobs = label(lis==1, connectivity=1)
pb = regionprops(blobs)
points = []
for parxy in pb:
     points.append([parxy.centroid[1], parxy.centroid[0]])
points = np.array(points)

#Ordenamos los puntos de la rama del rulo
#FutureWarning: Pass n_neighbors=2 as keyword args. From version 1.0 (renaming of 0.25) passing these as positional arguments will result in an error "will result in an error", FutureWarning)
clf = NearestNeighbors(2).fit(points)
G = clf.kneighbors_graph()
T = nx.from_scipy_sparse_matrix(G)

order = list(nx.dfs_preorder_nodes(T, 0))

x = points[:, 0]
y = points[:, 1]
xx = x[order]
yy = y[order]

paths = [list(nx.dfs_preorder_nodes(T, i)) for i in range(len(points))]

mindist = np.inf
minidx = 0

for i in range(len(points)):
    p = paths[i]           #order of nodes
    ordered = points[p]    #ordered nodes
    #find cost of that order by the sum of euclidean distances between points (i) and (i+1)
    cost = (((ordered[:-1] - ordered[1:])**2).sum(1)).sum()
    if cost < mindist:
        mindist = cost
        minidx = i

opt_order = paths[minidx]
xx = x[opt_order]
yy = y[opt_order]

#Interpolamos
t = np.arange(0, len(xx))
csx = CubicSpline(t, xx)
csy = CubicSpline(t, yy)
tn = np.linspace(t[0], t[-1], endpoint=True, num=50*len(xx))

