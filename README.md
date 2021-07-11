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
* _whos_: da información de lo que tenemos definido en el código hasta el momento.
* _plt.imshow_: muetra la imagen.
* _plt.colorbar_: agrega barra de intensidad de colores.
* _plt.set\_camp(gray)_: muestra la imagen en blanco y negro.

## Splines
Vamos a necesitar el paquete _scipy.interpolate_. 
Para hacer splines temporalmente (es decir en 1D) usamos el comando _splrep_ al que se le da un valor de x e y, devolviendo los parámetros del spline (no se bien que es exactamente lo que devuelve). Luego con la función _splev_, dándole un x y lo devuelto por _splrep_ podemos graficar el spline. _splev_ devuelve dos lista, la primera para el eje x y la segunda para el y.

Para 2D usamos _splprep_ donde también le pasamos x e y. También agregarle que el parámetro s=0 (asi pareciera funcionar mejor). Es devuelve lo mismo que antes más un array que funciona como el parámetro por el que corre la curva (que pasa por los puntos que pedí que haga splines). Para graficar vuelvo a usar _splev_.

Para sacar los puntos de la fibra hago un doble loop sobre la imagen de _thin(fibra)_ y me guardo los puntos en que sea _True_. Con esos tengo los puntos de la fibra. El problema es que no están ordenados de forma tal que pueda hacer splines, ya que necesito que los vecinos sean contiguos. 

## Update Splines
Estuvimos probando códigos para interpolar la fibra con splines cúbicos. Al momento tenemos dos: uno que supones que anda pero creemos que puede tardar mucho y no ser generalizable, otro que no anda pero es posiblemente más generalizable y un poco más óptimo.

Trabajamos por separado, cada uno propuso un código distinto.

Las preguntas que nos surgieron al trabajar de esta forma son:

* ¿Sacar puntos de la fibra es válido?
* ¿Está bien pasar dos veces por el mismo tramo?
* ¿Vale la pena buscar una forma de fusionar ambas formas? ¿O mejor usamos una sola u otra totalmente distinta?
* ¿Cómo afecta que la curva que nos queda es escalonada?
* ¿Sería buena idea hacer un fitting con los splines, en vez de interpolar con los splines?
* ¿Podríamos fijar un día a la semana para hablar?



