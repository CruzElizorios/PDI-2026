import cv2 
import numpy as np
import matplotlib.pyplot as plt

"""Problema 1 - Ecualización local de histograma 
La técnica de ecualización del histograma se puede extender para un análisis local, es decir, 
se puede realizar una ecualización local del histograma. El procedimiento consiste en 
definir una ventana cuadrada/rectangular (vecindario MxN) y mover el centro de la misma 
de píxel en píxel. En cada ubicación, se calcula el histograma de los puntos dentro de la 
ventana “deslizante” y se obtiene de esta manera, una transformación local de ecualización 
del histograma. Esta transformación se utiliza finalmente para mapear el nivel de intensidad 
del píxel centrado en la ventana bajo análisis, obteniendo así el valor del píxel 
correspondiente a la imagen procesada. Luego, se desplaza la ventana un píxel hacia el 
costado y se repite el procedimiento hasta recorrer toda la imagen. 
Esta técnica resulta útil cuando existen diferentes zonas de una imagen que poseen 
detalles, los cuales se quiere resaltar, y los mismos poseen valores de intensidad muy 
parecidos al valor del fondo local de la misma. En estos casos, una ecualización global del 
histograma no daría buenos resultados, ya que se pierde la localidad del análisis al calcular 
el histograma utilizando todos los píxeles de la imagen. 
Desarrolle una función para implementar la ecualización local del histograma, que reciba 
como parámetros de entrada la imagen a procesar, y el tamaño de la ventana de 
procesamiento (MxN). Utilice dicha función para analizar la imagen que se muestra en la 
Figura 1 (archivo Imagen_con_objetos_ocultos.tiff) e informe cuáles son los detalles 
escondidos en las diferentes zonas de la misma. Luego, desarrolle un análisis sobre la 
influencia del tamaño de la ventana en los resultados obtenidos. 
AYUDA: Con la función cv2.copyMakeBorder(img, top, bottom, left, right, borderType), 
puede agregar una cantidad fija de píxeles a una imagen, donde top, bottom, left y right son 
valores enteros que definen la cantidad de píxeles a agregar arriba, abajo, a la izquierda y a 
la derecha, respectivamente. Por otro lado, borderType define el valor a utilizar. Por ejemplo, 
si se utiliza borderType = cv2.BORDER_REPLICATE, se replicará el valor de los bordes."""


def ecualizacion_local(img, M, N):
    """Ecualizacion local de histograma utilizando ventana flotante MxN
    
        Parametros:
            img: np.array -> imagen en escala de grises (uint8)
            M: altura -> filas
            N: ancho -> columnas
    
    Retorna:
        imagen procesada
    """
    #cuantos pixeles por lado
    top    = M // 2
    bottom = M // 2
    left   = N // 2
    right  = N // 2

    #Bordes repitiendo el valor del pixel
    img_pad = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)

    # dimensiones imagen
    filas, columnas = img.shape

    #creo imagen de salida 
    img_out = np.zeros(img.shape, dtype=np.uint8)


    for i in range(filas):
        for j in range(columnas):

            # Extraer la ventana MxN centrada en el píxel (i, j)
            ventana = img_pad[i : i + M, j : j + N]

            # --- Ecualizar el histograma de la ventana (subimagen) -> (CDF) Función de Distribución Acumulada  ---
            ventana_heq = cv2.equalizeHist(ventana)

            # --- Tomar el píxel central de la ventana ecualizada ---
            img_out[i, j] = ventana_heq[top, left]

    return img_out


img = cv2.imread('TP_1/Imagen_con_detalles_escondidos.tif', cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap='gray'), plt.show(block=False)

# Ecualizacion global
img_global = cv2.equalizeHist(img)

# --- Ecualización local con distintos tamaños de ventana ---
ventanas = [
    (7,   7),
    (21, 21),
    (31, 31),
    (51, 51),
]

# ecualizacion local de la imagen con distintos tamanios de ventanas deslizantes.
resultados = []
for M, N  in ventanas:
    res = ecualizacion_local(img, M, N)
    resultados.append((res))

plt.figure(figsize=(16, 10))
plt.subplot(2, 3, 1); plt.imshow(img, cmap='gray'); plt.title('Original'); plt.axis('off')
plt.subplot(2, 3, 2); plt.imshow(img_global, cmap='gray'); plt.title('Global'); plt.axis('off')
plt.subplot(2, 3, 3); plt.imshow(resultados[0], cmap='gray'); plt.title('Local 7x7'); plt.axis('off')
plt.subplot(2, 3, 4); plt.imshow(resultados[1], cmap='gray'); plt.title('Local 21x21'); plt.axis('off')
plt.subplot(2, 3, 5); plt.imshow(resultados[2], cmap='gray'); plt.title('Local 31x31'); plt.axis('off')
plt.subplot(2, 3, 6); plt.imshow(resultados[3], cmap='gray'); plt.title('Local 51x51'); plt.axis('off')
plt.show()


#Filtro de la mediana para suavizar ruido
resultados = []
for M, N in ventanas:
    res = ecualizacion_local(img, M, N)
    res_filtrado = cv2.medianBlur(res, 3)
    resultados.append(res_filtrado)

plt.figure(figsize=(16, 10))
plt.subplot(2, 3, 1); plt.imshow(img, cmap='gray'); plt.title('Original'); plt.axis('off')
plt.subplot(2, 3, 2); plt.imshow(img_global, cmap='gray'); plt.title('Global'); plt.axis('off')
plt.subplot(2, 3, 3); plt.imshow(resultados[0], cmap='gray'); plt.title('Local 7x7'); plt.axis('off')
plt.subplot(2, 3, 4); plt.imshow(resultados[1], cmap='gray'); plt.title('Local 21x21'); plt.axis('off')
plt.subplot(2, 3, 5); plt.imshow(resultados[2], cmap='gray'); plt.title('Local 31x31'); plt.axis('off')
plt.subplot(2, 3, 6); plt.imshow(resultados[3], cmap='gray'); plt.title('Local 51x51'); plt.axis('off')
plt.show()

print("""Detalles ocultos 
      - sup.izq: cuadrado 
      - sup.der: línea diagonal 
      - centro: letra 'a' 
      - inf.izq: barras horizontales 
      - inf.der: círculo

Influencia del tamaño de ventana: ventanas pequeñas (7x7) revelan todos los detalles pero generan mayor cantidad de ruido;
ventanas grandes (31x31, 51x51) reducen el ruido pero pierden especificidad local. 
Conclusion: la ventana optima depende del tamaño del objeto a identificar.
""")


