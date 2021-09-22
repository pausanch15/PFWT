#Armo las figuras para los posters
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.measure import label, regionprops
from skimage.morphology import thin
plt.ion()

#%%
image = Image.open('imagenprueba.png') #cargo la imagen
ima = np.asarray(image) #la convierto en numpy array
im = ima[:,:,0] #eligo quedarme con el color rojo
imc = im[:, 100:800] #me quedo entre las lineas 100 y 800 (quito las turbinas)
imt = imc<200 #pongo el threshold en 200 para binarizar la imagen
li = label(imt) # etiqueto cada región contigua de pixeles llenos 
fibra = li==6  #me quedo con la fibra (vimos graficamente que la fibra tenia la label 6)
fibra = thin(fibra)
plt.imshow(fibra)
plt.colorbar() #para que muestre el rango en el que van los colores
plt.set_cmap('gray') #para que la imagen sea en blanco y negro
plt.show()

#%%
plt.rc("text", usetex=True)
plt.figure()
plt.plot([324, 297], [427, 566], '.', color='#ffc000', markersize=4, label='Bordes')
plt.plot([462, 474], [494, 491], '.', color='#70ad47', markersize=4, label='Nudos')
plt.xticks([])
plt.yticks([])
plt.imshow(fibra)
plt.colorbar() #para que muestre el rango en el que van los colores
plt.set_cmap('gray_r') #para que la imagen sea en blanco y negro
plt.legend()
plt.show()
plt.savefig('bordes_nudos.pdf', dpi=300)
