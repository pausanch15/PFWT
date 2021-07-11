# PFWT
Laboratorio 6 y 7

## Procesado de Imágenes
Partimos de una imagen. Hay que detectar los puntos de la fibra, especialmente sus extremos.

A los costados de la imagen se ve el setup del experimento.

Buena iluminación baja el ruido.

Las cámaras suelen captar mejor el verde. Esto igualmente puede variar entre distintas cámaras.

Distintos paquetes para cargar la imagen, usamos PIL (from PIL import Image).

RGBA: red/green/blue/alpha (transparencia).

Los píxeles que forman la imagen pueden tomar valores de intensidad entre 0 y 255. Los tonos más claros son píxeles cercanos a 255, los más oscuros se acercan a 0.

Python no necesariamente respeta la relación de aspecto de la imagen original, se puede pedir que lo haga. También da vuelta la imagen por como se toman las coordenadas, en la imagen el origen está arriba a la izquierda, en python esta abajo a la izquierda.

Podemos recortar la imagen para quedarnos con la zona donde puede aparecer una fibra.

Para procesar la imagen usamos _skimage_. Trabajamos con un solo color (o la suma de los colores también es valido). Si queres hacer otra cosa con la imagen puede variar que que/como usar esto.

Una vez tenemos lo anterior binarizamos la imagen con algún threshold (se puede hacer a ojo aunque hay mejores métodos). Luego usamos el comando de _label_ para dividir en distintas regiones. Con _regionprops_ podemos ver propiedades de cada región, la superficie/area es buen criterio para separar ruido de la fibra. Finalmente usamos el comando _thin_ para adelgazar la fibra, que sea de un píxel de ancho (es decir, al sacar un píxel se corta el dos regiones). 

**ROI**: Region of Interest.

Algunos comandos útiles que seguramente usemos:

    _whos_: da información de lo que tenemos definido en el código hasta el momento.
    _plt.imshow_: muetra la imagen.
    _plt.colorbar_: agrega barra de intensidad de colores.
    _plt.set\_camp(gray)_: muestra la imagen en blanco y negro.



