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

Hablamos con Pablo, pudimos responder las preguntas anteriores.

* ¿Sacar puntos de la fibra es válido? Si, no es un problema. No necesitamos interpolar rigurosamente, sino que queremos realizar un ajuste por splines, y sacar puntos es incluso parte del proceso.
* ¿Está bien pasar dos veces por el mismo tramo? Sí. Al ser todo ficticio lo que trabajamos con la fibra (distinto a lo que hace la fibra realmente) no es importante esto. Seguro estamos haciendo cosas similares en otros tramos sin saberlo. Además está bien por lo que explicó en la respuesta anterior.
* ¿Vale la pena buscar una forma de fusionar ambas formas? ¿O mejor usamos una sola u otra totalmente distinta? Sí vale la pena. Para hablar más de esto, leer más abajo las propuestas para el trabajo que sigue.
* ¿Cómo afecta que la curva que nos queda es escalonada? Esto hay que suavizarlo. Es por esto que no vamos a trabajar con una interpolación propiamente dicha sino con un fitting.
* ¿Sería buena idea hacer un fitting con los splines, en vez de interpolar con los splines? Sí! De hecho, es lo que deberíamos haber hecho desde un inicio, solo que como el fitting que deberíamos usar también se llama splines cúbicos lo entendimos mal. Más abajo describimos más los tipos de ajuste con los que podremos trabajar.
* ¿Podríamos fijar un día a la semana para hablar? Sí! Quedamos para el 4/5/21.

A la hora de trabajar con la imagen de la fibra, importa en que orden la recorremos ya que la energía depende del radio de curvatura y esto depende de cómo tomemos la fibra.

Fijamos un plan de trabajo a seguir. Lo que deberíamos lograr hacer para la semana que viene es lo siguiente:

* Detectar extremos y nudos.
* Partir la fibra (de la forma más genérica posible, con la idea en mente de que esto sirva para cualquier otra fibra del estilo de las que nos vamos a encontrar).
* Ordenar los puntos en cada tramo.
* Interpolar (en el sentido físico).
* Unir los pedazos en los que dividimos la fibra y sus respectivas interpolaciones.

En general buscaremos que el código use la menor cantidad de iterables posible, ya sean listas o arrays que tengamos que recorrer. Queremos evitar esto ya que es costoso y existen herramientas muy desarrolladas para evitar tener que hacerlo.

En cuanto a cómo partir la fibra, todo se reduce a poder encontrar los nudos (knots). Para esto pensamos en la imagen de la fibra que tenemos como una matriz donde cada lugar de la misma representa la intensidad de un píxel. Por otro lado, tomamos una matriz de dimensiones mucho más chicas que de la matriz de la imagen. A esta matriz la llamaremos _stencil_ o _kernel_, y con ella iremos recorriendo la imagen, colocando el stencil en alguna posición inicial y recorriendo de esta forma cada una de las posiciones, realizando a cada paso la convolución entre las matrices superpuestas (stencil convolucionado con la sección de la matriz de la imagen que corresponda). Por cada convolución obtendremos un valor, y en total tendremos tantos valores como píxeles en la imagen original. A la matriz de los productos de las convoluciones la llamaremos _imagen convolucionada_, y es la que usaremos para detectar los knots (y los end points). Los píxeles más claros de la imagen convolucionada corresponderán a los extremos (end points) de la fibra, mientras que los más opscuros serán los knots. Una función que hace esto en python es _miss\and\match_.

Esta es una forma mucho más óptima de obtener los knots y los endpoints que recorrer todos los puntos de la fibra ya que se trata de productos entre matrices chicas, operación que está muy optimizada en python. Paquetes que realizan esto de manera adecuada son _Lapack_ y _Blas_ entre otros.

A la hora de ordenar los puntos será conveniente pensar a la fibra como un grafo, y de esta manera contar con las nociones de _vecinos_ o _neighbours_ y _nodos_. Además, de esta forma nos podremos valer de rutinas _KNN_ (Kth neighbours), funciones ya desarrolladas y optimizadas para encontrar los caminos más cortos posibles para recorrer grafos pasando solamente una vez por cada nodo. De esta forma es que podremos ordenar los puntos de la fibra sin pasar por listas ni arrays.

En cuanto a la interpolación, no nos referimos a ella en el sentido más matemático, sino que lo que buscamos es un fitting a los puntos de la fibra mediante splines cúbicos. No tendrá sentido trabajar con cuadrados mínimos ya que muchas de las secciones de fibra que con encontremos no corresponderán a funciones (darán vueltas, serán rulos. etc).

Hay tres métodos que posiblemente usemos, y cada uno cuenta con un parámetro que tendremos que determinar de manera arbitraria, y representará el criterio con el que trabajemos.

* _Splines Cúbicos_: Depende del _subsampling_, es decir cuantos puntos tomar para hacer el fitting/interpolación. Entendí algo sobre tomar algún parámetro para usar como guía de que si saco más puntos la curva cambia demasiado.
* _Filtros Savitzky-Golay_: Dependen del tamaño de la ventana. Lo que hace es ir tomando promedios por ventana y asignarle ese valor a algún lugar y luego mover la ventana es como una media móvil.
* _Filtros en Fourier_: Dependen del _cut off_, de la frecuencia de corte. Básicamente es un pasabajos, dejamos pasar todas las frecuencias menores a el _cut off_.

Para unir las secciones en las que separamos la fibra, lo que nos convendrá hacer es extrapolar las aproximaciones por splines que hicimos, y de esta forma ir probando cómo serían todas las posibles combinaciones entre secciones, y compararlas para ver cuál genera menos nuevas curvaturas. Esto lo haremos tomando la extrapolación de cada spline con el que contemos, midiendo su curvatura en función de algún parámetro (en este caso la posición de cada punto). Para cada una de estas curvaturas tomaremos su media y desvío estándar. Luego, pegaremos cada par posible de extremos y observaremos los eventos que se den en cada caso. Cuando estos sean mucho mayores al desvío estándar observado en cada caso, sabremos que esa no es la forma de pegar las secciones.

Para unir la fibra Pablo sugirió la opción de ir uniendo forzosamente los tramos y ver que da menos curvatura. Más específicamente queremos hacer un gráfico de la curvatura en función del parámetro de la fibra y buscar que en los puntos donde se unan los tramos no tener picos muy altos (que se dispare el valor de la curvatura).

## Cómo guardar los Datos
Estamos viendo la mejor manera de almacenar los datos que vayamos obteniendo al analizar las imágenes que provengan de las mediciones. Esperamos tener bastantes datos, separados en diversas categorías. La idea sería usar _hdf5_, con la librería [**h5py**](https://docs.h5py.org/en/stable/) para poder trabajar desde python.

## Análisis de Imágenes: Buscando la Fibra
Estos meses estuvimos trabajando en los códigos para automatizar lo más posible el proceso de interpolar las imágenes de las fibras. Trabajamos en un código que simula una serie de imágenes con ruido, similares a las que esperamos obtener al medir en el laboratorio. Luego aplicamos los códigos anteriores a estas imágenes simuladas. Todos estos trabajos están en este repositorio de github.

Para generar fibras que se muevan aleatoriamente, lo que hacemos es tomar cuatro puntos aleatorios, unirlos mediante splines cúbicos, permitirles a estos puntos moverse cada uno con una random walk y volver a unir los puntos resultantes mediante splines nuevamente. Una vez que tenemos una serie de imágenes que sigen esta dinámica, medimos la longitud de la primer curva y forzamos a que el resto tenga la misma (cortando los extremos). Para evitar que una fibra difiera mucho respecto de la anterior, permitimos que la random walk se de en un radio determinao alrededor de cada punto. Esta fibra luego es colocada en un array de dos dimensiones, y de esta forma generamos la imagen a la cual le agregamos un gradiente de fondo, ruido del tipo _salt&pepper_ y una filtro gaussiano, ya que esperamos que las imágenes obtenidas en el laboratori vengan con este tipo de dificultades a la hora de analizarlas

Para encontrar la fibra en cada imagen, diferenciarla del ruido de fondo y poder pasarla por el algoritmo que la interpola, usamos la función de la librería **skimage**: [_remove_small_objects_](https://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.remove_small_objects). De esta forma, basta con pedirle a la función que quite del array original todos aquellos objetos de menor tamaño que la fibra, y así nos deshacemos del ruido.

Tenemos charla de avance el 3/6, y justamente esto que explicamos en el párrafo anterior es lo que vamos a contar.

## Análisis de Imágenes: Seguimos Buscando la Fibra
Tras probar diversas formas buscar los bordes y los knots, finalmente decidimos pegarnos al plan original, usando sténciles como describimos en secciones anteriores. Para ordenar los puntos de cada tramo de la fibra también seguimos la idea original: tratar a cada tramo como un grafo inconexo y usar la librería  [**networkx**](https://pelegm-networkx.readthedocs.io/en/latest/) para hallar el camino más corto posible que pase por todos los vecinos.

En cuanto a la forma en que simulamos las imágenes de las fibras con ruido, Pablo opinó que la diferencia entre un estado de la fibra y otro no era lo suficientemente marcada, por lo que carecería de sentido realizar estadística sobre un número grande de estas imágenes para hablar del error en nuestro algoritmo (diferencia entre la fibra original simulada y la que recuperamos tras todo el proceso de encontrar la fibra, dividirla en tramos, ordenar cada uno, interpolarlos y unirlos nuevamente). Además de que el código tardaba mucho en simular un número grande de imágenes. Ya en generar unas 30 tardaba más de dos segundos. Es por esto que nos sugirió generar fibras simuladas de la siguiente manera: crear una serie de cantidad de términos regulable, combinación lineal de senos y cosenos de diversas frecuencias y amplitudes para dar origen a los puntos de las fibras simuladas. Generar dos o más de estas fibras, y dar la dinámica entre ambos estadíos permitiendo que un punto se mueva desde su posición en la imagen inicial a su posición en la imagen final mediante un MRU. Esto nos permite regular cuánto queremos que la fibra se cruce con sigo misma, qué tan sinuosa es (agregando o quitando términos de la serie) y el número de puntos que posee en cada uno de los estadíos intermedios de la fibra dinámica.

Comenzamos a trabajar también con distintos **profilers** para identificar cuánto tarda el código en realizar cada acción, y de esta forma optimizar su funcionamiento. Esto es necesario ya que esperamos trabajar con una gran cantidad de imágenes extraidas del laboratorio, por lo que no podemos tener un código que tarde mucho en analizar cada una. Haciendo uso del **line_profiler** nos dimos cuenta de que lo que más tiempo lleva de todo el análisis es afinar la fibra mediante el uso de la función [_thin_](https://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.thin) una vez que la misma fue extraída del ruido de fondo de la imagen. Pablo propuso que el problema estaba en usar esta función en toda la imagen, y no solamente en la ROI. Pudimos cambiar esto. Usango la función [_regionprops_ ](https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_regionprops.html) pudimos facilmente quedarnos solamente con aquellos puntos de la imagen que corresponden a la fibra, hacer el thin sobre ellos y luego continuar con la imagen completa.

## Estadística y Validación
Teniendo el código funcionando, prodedemos a probar el algoritmo diseñado para detectar e interpolar las fibras. Para eso, generamos 3000 fibras aleatorias como explicamos antes, y nos guardamos los splines de las fibras originadas, para después compararlas con las reconstruidas. Lo hacemos de a una imagen a la vez: generamos la imágen, guardamos los splines de la fibra original y la imagen, analizamos con nuestro algoritmo, y nos guardamos los splines resultantes de la interpolación (en distintos archivos HDF5). Luego borramos de las variables del código estos elementos, así evitamos que se llene la memoria de la pc a medida que avanzamos en el número de imágenes.

Una vez obtenidos estos archivos,procedemos a evaluar los splines (originales y recuperados) en un array de 1000 puntos. Esto da como resultado arrays con puntos no equiespaciados, lo cual es un problema a la hora de comparar los datos originales con los obtenidos. Para solucionar esto, separamos en los datos correspondientes a las coordenadas x e y, y equiespaciamos cada tira de datos. Para eso hacemos uso de la función hallada en este [_link_](https://stackoverflow.com/questions/34941799/querying-points-on-a-3d-spline-at-specific-parametric-values-in-python). De esta manera podemos restar los x obtenidos a los originales (lo mismo para la coordenada y) y obtener una medida de la diferencia entre la curva original y la recuperada. Para etsas diferencias punto a punto, hacemos el histograma de las 3000 imágenes.

Lo que realmente nos interesa es poder calcular la curvatura de las fibras (mejor dicho, de las fibras reconstruidas) y obtener un resultado confiable para esta cantidad. Es por eso que repetimos la idea anterior pero con la curvatura. Una vez que contamos con los splines originales y recuperados evaluados, y en arrays equiespaciados, a cada una de estas fibras le calculamos la curvatura. Luego hacemos el histograma para la diferencia entre ambas curvaturas normalizada por el módulo de su suma.

Para realizar toda esta estadística, solo tuvimos en cuenta aquellas fibras cuya diferencia punto a punto respecto de la original no era mayor a 5 píxeles en ningún tramo. Solo encontramos que las fibras en forma de P presentaban diferencias tales. Queda como idea a futuro solucionar esto. De todas formas, no esperamos encontrar fibras que se doblen de tal manera en el laboratorio.

## Trabajo en el Laboratorio
Fuimos al laboratorio y nos encontramos con el set up de particle tracking y FTP montado, pero con algunos problemas en la cuba de agua. Para aumentar el contraste entre las partículas y el fondo, se usa dióxido de titanio, el cual había quedado disuelto en el agua de experimentos anteriores, por lo que se habían formado grumos y a decantar. Dado que Pablo se va de viaje por dos semanas en las que no vamos a ir al labo, no tuvo mucho sentido limpiar a fondo la cuba, por lo que realizamos unas primeras mediciones en estas condiciones., y aprovechamos a familiarizarnos con los equipos.

La cuba de acrílico cuenta con dos paletas acrílicas, conectadas a motores móviles los cuales se controlan desde la computadora. A los mismos se les pueden pasar funciones desde la pc, para que se muevan según ellas.

La cámara de alta resolución también se controla desde la computadora, y podemos elegir el número de frames por segundo que captura, y aplicar un shading, que nos permite fijar el valor de intensidad que la cámara tomará como negro. Tiene más funciones pero no las usaremos nosotros. 

Algo llamativo de la cámara es que la misma está permanentemente grabando. Guardará el número de imágenes que pueda hasta llenar la memoria, dado el número de frames por segundos que se setee. Una vez que la memoria se llena, comienza a sobreescribir, pero nunca deja de grabar. De esta manera, podemos hacer que comience una medición (imágenes que no va a sobrescribir) eligiendo nosotros el instante inicial en el cuál empezamos a grabar, el medio o el final. Si inicialmente elegimos que queremos tener n imágenes, con esta función de la cámara podemos quedarnos con las n posteriores al momento en el que hacemos click, con n/2 antes y después del momento en el que hacemos click o con las n previas.

Si bien no pusimos a punto la cuba de agua, la limpiamos un poco y le agregamos agua (de la canilla), y comenzamos a hacer algunas mediciones preliminares. Para FTP usamos distintos patrones rayados, y uno gris. Lo que sucedió fue que aunque le agregáramos dióxido de titanio, el mismo decantaba dado que ya existían grumos previos y no se podía disolver del todo, por lo que el contraste en el patrón no fue el óptimo. De todas formas, tomamos unas 300 imágenes del patrón quieto, para usar de referencia. También tomamos unas 3000 imágenes del patrón con el agua en movimiento, usando los motores que dijimos al principio. 

Luego hicimos unas pruebas con una fibra que no cumple con lo que buscamos nosotros, pero fueron mediciones de prueba. Usamos un pelo de un pincel común, demasiado rígido. Si bien pudimos obtener unas 1000 imágenes de esta fibra, no fueron óptimas ya que el pelo se hundía (creemos que por la rara densidad que tendría el agua dada la cantidad de dióxido de titanio mal disuelto).

Para llevarnos todas estas imágenes, las subimos a la nube del df y posteriormente las descargaremos en casa.

## Análisis Imágenes de Prueba
El miércoles pudimos sacar alguna imágenes preliminares (ya que la solución de agua con TiO no era óptima, ni tampoco las fibras) y estamos tratando de analizarlas. Hasta ahora se analizó el video con la fibra. 

Lo primero a notar es que para poder binarizar la imagen hay que restar el promedio de imágenes de grises, para suavizar un poco el fondo. También hay una punta de la fibra se desvanece de a poco de la imagen. Esto puede ser porque se encontraba hundida en el agua. Esto habría que solucionarlo de alguna forma, ya que vuelve más complicado reconocer la fibra, la cual siempre queda cortada. 

El mayor reto fue poder distinguir la fibra del fondo en cada imagen, ya que el valor necesario para binarizar cambia en cada imagen. Tomar como valor de binarización el valor medio de la imagen menos 2.2 veces la desviación estándar ($\mu - 2.2 \sigma$) funciona para distinguir la fibra en casi todos los casos, pero esto hace que parte del ruido también se mezcle. 

Para distinguirlos la idea es usar la función _label_ junto con _regionprops_ para sacar algún valor que permita distinguir la fibra. Para estas imágenes encontré tres posibles propiedades que pueden ser utiles (en cursiva la descripción de _regionprops_):
    * _Excentricidad_: _Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance
    (distance between focal points) over the major axis length. he value is in the interval [0, 1). When it is 0, the ellipse becomes a circle._ Para esta fibra funcionó muy bien, permite distinguir la región de la fibra de los ruidos casi siempre, dado que daba un valor de 0.999 aprox. Sin embargo, es posible que esto sea porque está fibra es muy recta y que no funcione muy bien para fibras más curvadas. Hay que probarlo
  * _Número de Euler_: _Euler characteristic of region. Computed as number of objects (= 1) subtracted by number of holes (8-connectivity)._ Para la fibra tuvo siempre un valor de 0 o 1. Habría que ver para fibra que se cruzan a si mismo si cambia. Aunque sea poco frecuente, a veces el ruido también puede dar un valor de 0 o 1. 
   * _Coeficientes de area_: No es una propiedad de _regionprops_. Es calcular el área de la \textit{bbox} divido el área de la región (sería: $bbox_area / area$, para cada región). Pareciera prometedor, aunque a veces puede ser un poco ajustado el margen entre fibra y ruido (por lo general no igual).

Una última observación es que a veces se mezcla un poco de ruido en la fibra, lo cual lleva a tener pequeñas "ramas" que se desprenden de lo que es la fibra. Esto a veces hace confundir al algoritmo, llevando a que la reconstrucción no sea lo que debe ser. (\textcolor{red}{UPDATE: pareciera que usar \textit{skeletonize} en vez de \textit{thin} resuelve esto último un poco. Sigue pasando pero mucho menos})

Pequeña tangente: encontre el llamado filtro sato (sato tubeness filter) que permitiría distinguir la fibra super fácil (además de que quedaría bastante suave), pero tarda mucho tiempo en correr. Para hacerlo viable habría que aplicarlo a una pequeña región donde se encuentre la fibra. Para encontrar esa región no sé que como habría que hacerlo para que usar este filtro tenga sentido.
