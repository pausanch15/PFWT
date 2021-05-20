# PFWT
Laboratprio 6 y 7

## Comentarios sobre las funciones en Splines.py
**cortar_fibra(fibra)**:
Resibe la fibra luego de hacerle el skeletonize (o el thin) y devuelve los una lista de listas con los tramos y otra con los bordes

**ordenar_fibra(tramos)**:
Resibe la lista de listas de tramos duvuelta por *cortar_fibra*, y devuelve una lista de listas pero con esos tramos ordenados 

**pegar_fibra(tramos,bordes,tamano_nudo=30, window=21, s=10)**:
Resibe los tramos devueltos por *ordenar_fibra* y los bordes de *cortar_fibra*. Devuelve el parametro de la curva (t), la curvatura de la curva en funcion del parametro (curvatur), y los puntos de la fibra suavizada (xf,yf). Nota: en el proceso de pegar los tramos suavizo la curva (para porder calcular bien la curvatura).  

**crea_fibra(n=4,bins=50)**:
Crea una imagen de una fibra haciendo splines de n puntos. No es controlable si hay o no nudo, ni cuantos (aunque no llegue a ver con más de 1). Conviene usar *np.random.seed(algun_numero)* para poder repetir la configuración que salga

***Algunos Problemas***

Si la fibra se divide en varias secciones (mas de 5-6 diría), como tengo una combinatoria, se dispara la cantidad de operaciones y el programa se traba. Para resolver esto me parece que habría que usar otra forma de pegar la fibra.

Otro porblema es si la fibra toma forma de P, es decir que solo hay un borde. Creo que podria resolverse facil con algunos *if* pero no tengo ganas de hacerlo ahora
