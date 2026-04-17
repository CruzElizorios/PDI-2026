import cv2
import numpy as np
import matplotlib.pyplot as plt




# import cv2
# import numpy as np

# # Carga de la imagen
# imagenGris = cv2.imread('tp1/examen_1.png', cv2.IMREAD_GRAYSCALE)

# # Binarización (Unidad 3: Otsu o Umbral Fijo)
# # Usamos THRESH_BINARY_INV para que las líneas sean blancas, por arriba de 150 se vuelven negras, por debajo de 150 se vuelven blancas
# _, imagenBinaria = cv2.threshold(imagenGris, 150, 255, cv2.THRESH_BINARY_INV)

# plt.imshow(imagenBinaria, cmap='gray')
# plt.title('imagenBinaria')
# plt.axis('off')
# plt.show()

# # Lineas Horizontales
# # Sumamos todos los píxeles de cada fila
# sumaHorizontal = np.sum(imagenBinaria, axis=1)

# # Buscamos dónde la suma supera un umbral (por ejemplo, el 50% del ancho de la imagen * 255)
# anchoImagen = imagenBinaria.shape[1]
# umbralLinea = anchoImagen * 255 * 0.5
# indicesLineasHorizontales = np.where(sumaHorizontal > umbralLinea)[0]
# print("Índices de líneas horizontales:", indicesLineasHorizontales)

# # Lineas Verticales
# # Sumamos todos los píxeles de cada columna
# sumaVertical = np.sum(imagenBinaria, axis=0)
# # Buscamos las líneas verticales
# altoImagen = imagenBinaria.shape[0]
# umbralLineaVertical = altoImagen * 255 * 0.3
# indicesLineasVerticales = np.where(sumaVertical > umbralLineaVertical)[0]
# print("Índices de líneas verticales:", indicesLineasVerticales)



import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Preprocesamiento de la Imagen
# =============================================================================
# Cargamos la imagen original (color para dibujar encima luego)
imagenOriginal = cv2.imread('tp1/examen_3.png')
imagenGris = cv2.cvtColor(imagenOriginal, cv2.COLOR_BGR2GRAY)

# Binarización Inversa (Unidad 3)
# El fondo negro (0) y lo que tiene "tinta" blanco (255)
_, imagenBinaria = cv2.threshold(imagenGris, 150, 255, cv2.THRESH_BINARY_INV)

# =============================================================================
# Perfiles de Proyección (Detección de Líneas)
# =============================================================================
# Calculamos la suma de píxeles blancos por fila y por columna (Unidad 1 - NumPy)
sumaFilas = np.sum(imagenBinaria, axis=1)
sumaColumnas = np.sum(imagenBinaria, axis=0)

# Definimos umbrales para identificar qué es una "línea larga"
# Una línea de la tabla debería ocupar gran parte del ancho/alto
umbralFila = imagenBinaria.shape[1] * 255 * 0.5    # 50% del ancho
umbralColumna = imagenBinaria.shape[0] * 255 * 0.5 # 20% del alto (las verticales son más cortas)

# Obtenemos los índices donde hay picos de intensidad
indicesFilas = np.where(sumaFilas > umbralFila)[0]
indicesColumnas = np.where(sumaColumnas > umbralColumna)[0]


# =============================================================================
# PASO 1.4: Limpieza y Obtención de Límites Únicos
# =============================================================================
def agruparIndicesLineas(indices, distanciaMinima=10):
    """
    Como las líneas tienen varios píxeles de ancho, esta función se queda 
    con un solo índice (el primero) de cada grupo de píxeles consecutivos.
    """
    if len(indices) == 0:
        return []
    
    limitesSimplificados = [indices[0]]
    for i in range(1, len(indices)):
        # Si la distancia entre este índice y el anterior es grande, es una nueva línea
        if indices[i] > indices[i-1] + distanciaMinima:
            limitesSimplificados.append(indices[i])
    return limitesSimplificados

filasUnicas = agruparIndicesLineas(indicesFilas)
columnasUnicas = agruparIndicesLineas(indicesColumnas)

# =============================================================================
# VISUALIZACIÓN DE LÍMITES DETECTADOS
# =============================================================================
imagenCopiaVisualizacion = imagenOriginal.copy()

# Dibujamos las líneas horizontales detectadas en Rojo
for fila in filasUnicas:
    cv2.line(imagenCopiaVisualizacion, (0, fila), (imagenOriginal.shape[1], fila), (0, 0, 255), 2)

# Dibujamos las líneas verticales detectadas en Azul
for col in columnasUnicas:
    cv2.line(imagenCopiaVisualizacion, (col, 0), (col, imagenOriginal.shape[0]), (255, 0, 0), 2)

# Mostramos resultados
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Imagen Binaria (Entrada del Algoritmo)")
plt.imshow(imagenBinaria, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Límites de Grilla Detectados")
plt.imshow(cv2.cvtColor(imagenCopiaVisualizacion, cv2.COLOR_BGR2RGB))
plt.show()

# Imprimimos para control
print(f"Líneas horizontales encontradas: {len(filasUnicas)}")
print(f"Líneas verticales encontradas: {len(columnasUnicas)}")

# --- Aquí terminaría la preparación de la grilla ---
# El siguiente paso (Punto A) será usar estos índices para recortar
# cada celda y analizar las Componentes Conectadas dentro.



# --- PASO 2: Análisis de Respuestas ---
# --- Vector de respuestas correctas y mapeo de letras ---
respuestasCorrectas = ['C', 'B', 'A', 'D', 'B', 'B', 'A', 'B', 'D', 'D']
letrasOpciones = ['A', 'B', 'C', 'D']

# Suponemos que filasUnicas y columnasUnicas ya contienen los límites de la tabla.
# La tabla suele ser la parte inferior de la imagen. Tomamos los últimos índices.
filasPreguntas = filasUnicas[-11:]   # 10 preguntas + línea de cierre
columnasOpciones = columnasUnicas[-5:] # 4 columnas (A,B,C,D) + línea de cierre

print("--- RESULTADOS DEL EXAMEN ---")

# Recorremos cada una de las 10 preguntas (filas)
for i in range(10):
    opcionDetectada = None
    contadorMarcas = 0
    
    # Límites Y de la fila actual
    ySuperior = filasPreguntas[i]
    yInferior = filasPreguntas[i+1]
    
    # Recorremos las 4 celdas de la fila (columnas A, B, C, D)
    for j in range(4):
        # Límites X de la celda actual
        xIzquierda = columnasOpciones[j]
        xDerecha = columnasOpciones[j+1]
        
        # --- RECORTE CON MARGEN (Unidad 1) ---
        # Aplicamos un margen para "meternos" dentro del recuadro negro
        margen = 8 
        celdaRecortada = imagenBinaria[ySuperior+margen : yInferior-margen, 
                                        xIzquierda+margen : xDerecha-margen]
        
        # --- DETECCIÓN DE CONTORNOS Y AGUJEROS (Unidad 3) ---
        # Usamos RETR_TREE para obtener la jerarquía (agujeros de las letras)
        contornos, jerarquia = cv2.findContours(celdaRecortada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtramos por área para evitar ruido (puntos o restos de la línea negra)
        areaMinimaLetra = 100
        letraEncontradaEnCelda = False
        
        for k, cnt in enumerate(contornos):
            area = cv2.contourArea(cnt)
            if area > areaMinimaLetra:
                # Si el contorno no tiene "padre", es un objeto principal (posible letra)
                if jerarquia[0][k][3] == -1:
                    letraEncontradaEnCelda = True
                    # Contamos cuántos "hijos" (agujeros) tiene este contorno
                    agujeros = 0
                    hijo = jerarquia[0][k][2]
                    while hijo != -1:
                        agujeros += 1
                        hijo = jerarquia[0][hijo][0] # Siguiente hermano del hijo
                    
                    # Lógica topológica para identificar la letra:
                    # B -> 2 agujeros | A/D -> 1 agujero | C -> 0 agujeros
                    # En este ejercicio, basta con saber si la celda 'j' tiene contenido.
                    opcionDetectada = letrasOpciones[j]
                    contadorMarcas += 1

    # --- VALIDACIÓN DE LA PREGUNTA ---
    # REGLA: Incorrecta si hay 0 marcas o más de 1 marca.
    if contadorMarcas == 1 and opcionDetectada == respuestasCorrectas[i]:
        estado = "OK"
    else:
        estado = "MAL"
        
    print(f"Pregunta {i+1}: {estado}")
















### NO UTILIZADO
import cv2
import numpy as np

# --- PASO 1. Cargar la imagen original ---
# Cargamos la imagen a color para poder dibujar líneas de colores brillantes encima.
imagenOriginal = cv2.imread('examen_3.png')
# Hacemos una copia para no modificar la original al dibujar.
imagenVisualizacion = imagenOriginal.copy()

# --- PASO 2. Preprocesamiento (Conversión y Binarización Inversa - Unidad 3) ---
# 1. Convertir a escala de grises.
imagenGris = cv2.cvtColor(imagenOriginal, cv2.COLOR_BGR2GRAY)
# 2. Aplicar un umbral inverso (Thresholding Binario Inverso).
# Esto es CRUCIAL: las líneas negras y marcas X deben quedar BLANCAS (255)
# y el papel NEGRO (0) para que el análisis de componentes funcione.
# Usamos un umbral fijo (ej. 150) o Otsu si la iluminación lo permite.
_, imagenBinaria = cv2.threshold(imagenGris, 150, 255, cv2.THRESH_BINARY_INV)

# --- PASO 3. Análisis de Componentes Conectadas con Estadísticas (Unidad 3) ---
# Esta función encuentra todas las manchas blancas y calcula sus datos.
# ltype=cv2.CV_32S se usa para imágenes grandes.
resultado = cv2.connectedComponentsWithStats(imagenBinaria, connectivity=8, ltype=cv2.CV_32S)
numEtiquetas, etiquetas, estadisticas, centroides = resultado

# --- PASO 4. Filtrado y Visualización de Límites ---
# Definimos los colores y grosor para dibujar los rectángulos (BGR).
colorLimites = (0, 255, 0) # Verde brillante para los rectángulos.
grosorLinea = 2

# Iteramos sobre todas las componentes detectadas.
# IMPORTANTE: Empezamos en i=1 porque la etiqueta i=0 es el FONDO negro.
for i in range(1, numEtiquetas):
    # Extraemos las estadísticas de la componente 'i': [x, y, width, height, area]
    x = estadisticas[i, cv2.CC_STAT_LEFT]
    y = estadisticas[i, cv2.CC_STAT_TOP]
    ancho = estadisticas[i, cv2.CC_STAT_WIDTH]
    alto = estadisticas[i, cv2.CC_STAT_HEIGHT]
    area = estadisticas[i, cv2.CC_STAT_AREA]

    # --- PISTA CRÍTICA DEL EJERCICIO: FILTRADO ---
    # La ayuda dice: "eliminar las componentes conectadas de área muy chica".
    # Además de las celdas, el algoritmo detectará letras individuales (Name, Date),
    # puntos sueltos y quizás líneas largas de la tabla.
    # Debes definir umbrales basados en el tamaño esperado de una celda de respuesta.

    # Ejemplo de umbrales estimados (estos valores deben calibrarse para tu imagen):
    areaMinimaEsperada = 50   # Ignora ruido muy pequeño.
    areaMaximaEsperada = 10000 # Ignora las líneas largas de la tabla.
    anchoMinimo = 10
    altoMinimo = 10

    # Verificamos si la componente cumple con los criterios de tamaño de una celda/letra.
    if (areaMinimaEsperada < area < areaMaximaEsperada) and \
       (ancho > anchoMinimo) and (alto > altoMinimo):

        # Si pasa el filtro, dibujamos el rectángulo límite (Bounding Box)
        # sobre la imagen a color de visualización.
        # cv2.rectangle(imagen, punto_inicio(x,y), punto_fin(x+w, y+h), color, grosor)
        cv2.rectangle(imagenVisualizacion, (x, y), (x + ancho, y + alto), colorLimites, grosorLinea)

# --- PASO 5. Mostrar el resultado ---
# Para comparar, mostramos la imagen binaria (lo que ve el algoritmo) y la de visualización.
cv2.imshow('1. Imagen Binaria (lo que ve el algoritmo)', imagenBinaria)
cv2.imshow('2. Límites Detectados (con filtrado)', imagenVisualizacion)

# Esperar a que se presione una tecla y cerrar ventanas.
cv2.waitKey(0)
cv2.destroyAllWindows()