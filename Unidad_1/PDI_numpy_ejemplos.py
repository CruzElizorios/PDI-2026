import numpy as np
# https://numpy.org/doc/stable/user/numpy-for-matlab-users.html

# --- Vectores y Matrices -----------------------------------------------------
# Definición de un vector
x = np.array([1,2,3])
x
type(x)
x.ndim
x.dtype         # Tipo de datos que contiene el vector
len(x)          # Cantidad de elementos del vector
x.shape         # Tupla con el tamaño del vector (notar el ",)")

# Esto sería una matriz de 1x3 (una dimensión "singleton")
x = np.array([[1,2,3]])
x
type(x)
x.dtype
x.shape
len(x)

# Especifico el tipo de dato
x = np.array([1,2,3], dtype=np.uint8)
x.dtype

# Definición de una Matriz
x = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]], dtype=np.uint8)
x
type(x)
x.ndim
x.dtype             # Tipo de datos que contiene la matriz
x.shape             # Tupla con el tamaño de cada dimensión de la matriz
h, w = x.shape      # Unpacking de una tupla --> h:height (alto) | w:width (ancho)
len(x)              # Devuelve la longitud de la 1er dimensión (filas en este caso)
x.T                 # Transpuesta de la matriz

# Indexado
x = np.array([[1,2,3],[4,5,6],[7,8,9]])
x[0,0]      
x[0:2,1:2]  # Esto es una matriz...
x[0:2,1]    # ... esto es un vector (ambos con los mismos elementos)
x[0:100,:]  
x[1]

x[1,:] = np.array([8,8,9])
x[:,1] = np.array([-1,-1,-1])
x[:,2] = [9,9,9]
x[0,:] = 99
x[0:2,0:2] = 77


# --- Operaciones sobre vectores ----------------------------------------------
x = np.array([1,2,3,0,6,9,7,2], dtype=np.uint8)
x
x.min()             # Valor mínimo del vector
x.max()             # Valor máximo del vector
x.argmin()          # Indice del valor mínimo del vector
x.argmax()          # Indice del valor máximo del vector
np.unique(x)        # Valores únicos del vector (valores sin repetición)
len(np.unique(x))
x.sum()

# Ordenamiento
x.sort()            # Modifica el vector!
x2 = x.copy()       # No lo...
x2.sort()           # ... modifico

ix = x.argsort()	# Indices que ordenan al vector
x3 = x[ix]			# Vector ordenado

x==2
np.where(x==2)		# Indices donde está el valor buscado.
a = np.where(x==2)


# --- Operaciones sobre matrices --------------------------------
x = np.array([[2,0,4],[1,6,9],[3,8,5],[12,7,11]], dtype=np.uint8)
x
x.min()     # Valor mínimo de la matriz
x.max()     # Valor máximo de la matriz
x.argmin()  # Indice lineal del valor mínimo de la matriz
x.argmax()  # Indice lineal del valor máximo de la matriz

x[x.argmax()]     # Por que da error?
# Opción 1: Indexo linealmente una matrix --> flat
x.flat
type(x.flat)
x.flat[x.argmax()]
# Opción 2: Transformo el índice lineal en indices de fila/columna donde se da el mínimo/máximo
r,c = np.unravel_index(x.argmax(), x.shape)
x[r, c]

np.unique(x)        # Valores únicos de la matriz (valores sin repetición)
x.sum()

# Analisis en una sola dimensión
#   axis=0  --> Analizo columnas
#   axis=1  --> Analizo filas
x
x.min(axis=0)
x.max(axis=0)
x.argmin(axis=0)
x.argmax(axis=0)
x.sum(0)    # Sumo sobre las filas
x.sum(1)    # Sumo sobre las columnas

# Ordenamiento
x.sort()
# np.sort(x)
x.sort(0)
x.argsort()
x.argsort(0)

# Ejemplo particular 
x = np.array([[1,50,0.1],
              [3,60,0.4],
              [5,15,0.9],
              [2,35,1.5],
              [0,22,7.2],
              [4,18,9.6]])

x[:,0].argsort()
x[x[:,0].argsort(),:]
x[x[:,0].argsort()]     # Es igual a lo anterio.... por que?

# --- Otros metodos/funciones útiles -------------------------------------------
x = np.arange(10)       # Genero un vector de 10 elementos, empezando en el valor x default (0)
x = np.arange(4, 9)     # Genero un vector que va de 4 a 8 (9-1) inclusive
x = np.arange(4, 9, 2)  # Genero un vector que va de 4 a 8 (9-1) inclusive con paso igual a 2

# --- Misceláneas --------------------------------------------------
np.zeros(6)
np.zeros((3,6))
np.zeros((3,6), dtype=np.uint8)
np.ones((3,3))
np.diag([1,2,3,4])
np.diag(np.arange(1,5))
np.diag([1,2,3,4],-1)

