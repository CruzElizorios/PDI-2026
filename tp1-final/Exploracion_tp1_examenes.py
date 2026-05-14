import cv2 
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('TP_1')
from Resolucion_tp1_examenes.py import corregir_examen, detectar_lineas, agrupar_lineas, extraer_celdas, validar_encabezado, detectar_letra

#importacion imagen
img = cv2.imread('tp1-final/examen_2.png', cv2.IMREAD_GRAYSCALE)
print(img.shape)
print(img.dtype)
print(f"Min: {img.min()}, Max: {img.max()}")
plt.imshow(img, cmap='gray'), plt.show(block=False)

#Umbralizacion
img_th = (img.copy() < 200).astype(np.uint8)

plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray');    plt.title('Original'); plt.axis('off')
plt.subplot(1, 2, 2); plt.imshow(img_th, cmap='gray'); plt.title('Umbralizada'); plt.axis('off')
plt.show()

#--------------------------------------------------------------------------DETECTAR LINEAS-------------------------------------------------------------------------

img_rows = np.sum(img_th, axis=1)  # suma horizontal (H,) → detecta líneas horizontales
img_cols = np.sum(img_th, axis=0)  # suma vertical   (,W) → detecta líneas verticales

# Detectamos umbrales en filas 
plt.figure()
plt.plot(img_rows, np.arange(len(img_rows)))
plt.axvline(350, color='red', linestyle='--', label='umbral actual')
plt.axvline(200, color='green', linestyle='--', label='umbral propuesto')
plt.legend()
plt.grid(True)
plt.gca().invert_yaxis()
plt.show()

# --- Filas ---
fig, (ax_img, ax_proj) = plt.subplots(1, 2, sharey=True, figsize=(14, 6))
ax_img.imshow(img, cmap='gray')
ax_img.set_title('Imagen Original')
ax_proj.plot(img_rows, np.arange(len(img_rows)))
ax_proj.set_title('Proyección filas')
ax_proj.set_xlabel('Suma de píxeles')
ax_proj.grid(True)
plt.tight_layout()
plt.show()

# --- Columnas ---
fig2, (ax_img2, ax_proj2) = plt.subplots(2, 1, sharex=True, figsize=(14, 6))
ax_img2.imshow(img, cmap='gray')
ax_img2.set_title('Imagen Original')
ax_proj2.plot(img_cols)
ax_proj2.set_title('Proyección columnas')
ax_proj2.set_ylabel('Suma de píxeles')
ax_proj2.grid(True)
plt.tight_layout()
plt.show()

#Umbrales para obtener lineas (filas y columnas) necesarias.
th_row = 290
th_col = 400

# Antes de agrupar (algunas lineas ocupan varios pixeles) el np.where agarrar runs de pixeles que debemos agrupar.
raw_h = np.where(img_rows > th_row)[0]
raw_v = np.where(img_cols > th_col)[0]
print(f"ANTES - h_lines: {raw_h}  -> {len(raw_h)}")
print(f"ANTES - v_lines: {raw_v}  -> {len(raw_v)}")

# Después de agrupar
filas = agrupar_lineas(raw_h, gap=10).astype(int)
columnas = agrupar_lineas(raw_v, gap=30).astype(int)
print(f"DESPUÉS - filas: {filas}  -> {len(filas)}")
print(f"DESPUÉS - columnas: {columnas}  -> {len(columnas)}")

# Identificar cajas 
fig, ax = plt.subplots(figsize=(8, 10))
ax.imshow(img, cmap='gray')
for y in filas:
    ax.axhline(y, color='red', linewidth=0.5, alpha=0.7)
for x in columnas:
    ax.axvline(x, color='blue', linewidth=0.5, alpha=0.7)
plt.title('Líneas detectadas')
plt.show(block=False)

print(f"filas: {filas}  -> {len(filas)}")
print(f"columnas: {columnas}  -> {len(columnas)}")

'''
# entre 0 y filas[1]=53 se encuentra el encabezado
# filas[0]=32, filas[1]=53 -> margen
# filas[1]=53 a filas[2]=180 -> fila de preguntas 1 y 6
# filas[2]=180 a filas[3]=305 -> fila de preguntas 2 y 7
# filas[3]=305 a filas[4]=431 -> fila de preguntas 3 y 8
# filas[4]=431 a filas[5]=558 -> fila de preguntas 4 y 9
# filas[5]=558 a filas[6]=683 -> fila de preguntas 5 y 10
# para columnas:
# columnas[0]=17 a columnas[1]=259 -> columna izquierda (preguntas 1-5)
# columnas[2]=322 a columnas[3]=565 -> columna derecha (preguntas 6-10)
'''
#-----------------------------------------------------------------------------EXTRAER CELDAS----------------------------------------------------------------------
#crop de celdas y encabezado
celdas = {}

#Agrupamos filas en boxes de preguntas
rango_filas = [(filas[1], filas[2]),   # pregunta 1 y 6
         (filas[2], filas[3]),   # pregunta 2 y 7
         (filas[3], filas[4]),   # pregunta 3 y 8
         (filas[4], filas[5]),   # pregunta 4 y 9
         (filas[5], filas[6])]   # pregunta 5 y 10

#Agrupamos columnas 
rango_cols = [(columnas[0], columnas[1]),    # columna izquierda
        (columnas[2], columnas[3])]    # columna derecha

#Crop de las celdas
for i, (y1, y2) in enumerate(rango_filas):
    for j, (x1, x2) in enumerate(rango_cols):
        nro = i + 1 if j == 0 else i + 6
        celdas[nro] = img[y1:y2, x1:x2]
    
#mostramos celdas
fig, axes = plt.subplots(2, 5, figsize=(18, 6))
for nro, celda in celdas.items():
    fila = 0 if nro <= 5 else 1
    col  = (nro - 1) % 5
    axes[fila, col].imshow(celda, cmap='gray')
    axes[fila, col].set_title(f'P{nro}')
    axes[fila, col].axis('off')
plt.suptitle('Celdas de preguntas')
plt.tight_layout()
plt.show()

#Hacemos el crop del encabezado
encabezado = img[0:int(filas[1]), int(columnas[0]):int(columnas[3])] 
print(encabezado.shape)
plt.figure(figsize=(10, 2))
plt.imshow(encabezado, cmap='gray')
plt.title('Encabezado')
plt.axis('off')
plt.show()

#Umbralizado del encabezado.
enc_th = (encabezado < 200).astype(np.uint8)

enc_cols = np.sum(enc_th, axis=0) #proyeccion de columnas

#graficamos para detectar donde comienzan y terminan las lineas donde esta el texto.
plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1); plt.imshow(encabezado, cmap='gray'); plt.axis('off')
plt.subplot(1, 2, 2); plt.plot(enc_cols); plt.grid(True); plt.title('Proyección columnas encabezado')
plt.tight_layout()
plt.show()

#-------------------------------------------------------------CROP ENCABEZADO------------------------------------------------------ 
enc_rows = np.sum(enc_th, axis=1)

#buscamos la linea horizontal mas larga
fila_linea = np.where(enc_rows > 200)[0][0]
fila = enc_th[fila_linea, :]

#Identificamos lo que abarca cada segmento de linea.
#agregamos un cero al principio y al final de cada transicion
diff_fila = np.diff(fila, prepend=0, append=0)
#
inicios_campos = np.where(diff_fila == 1)[0]
fines_campos   = np.where(diff_fila == -1)[0]


#--------------------crop de las etiquetas del encabezado-------------------------------------------
campo_name  = encabezado[:, inicios_campos[0]:fines_campos[0]]
campo_date  = encabezado[:, inicios_campos[1]:fines_campos[1]]
campo_class = encabezado[:, inicios_campos[2]:fines_campos[2]]

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
axes[0].imshow(campo_name,  cmap='gray'); axes[0].set_title('Name');  axes[0].axis('off')
axes[1].imshow(campo_date,  cmap='gray'); axes[1].set_title('Date');  axes[1].axis('off')
axes[2].imshow(campo_class, cmap='gray'); axes[2].set_title('Class'); axes[2].axis('off')
plt.tight_layout()
plt.show()

#----------------------Umbralizacion para deteccion de nombre
campo_name_th = (campo_name < 200).astype(np.uint8)

# Detectar componentes conectadas (grupos de píxeles conectados = caracteres)
# Devuelve: cantidad de componentes, imagen etiquetada, estadísticas (x,y,w,h,area), centroides
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(campo_name_th, 8, cv2.CV_32S)
'''-------------------------------------------HASTA ACA LLEGE CON LOS COMENTARIOS----------------------------------------------'''
# imprimimos cantidad de componentes
print(f"num_labels: {num_labels}")
# Representacion de valores: (x, y, w, h, area) por cada fila (componente)

# ??????????????
print(f"stats:\n{stats}")

#filtramos por area (quitamos fondo y fin de linea) ???????????????
stats_filtradas = stats[(stats[:, 4] > 20) & (stats[:, 4] < 100)]
print(f"Componentes válidas: {len(stats_filtradas)}")
print(stats_filtradas)

# identificamos gaps entre letras (para distinguir palabras)
posiciones_x = stats_filtradas[:, 0]
gaps = np.diff(posiciones_x)
print(gaps)

#Espacio que marca una division de palabras
umbral_espacio = 15
n_espacios = np.sum(gaps > umbral_espacio)
n_palabras = n_espacios + 1

if len(stats_filtradas) == 0:
    print("Name: MAL")
elif n_palabras < 2:
    print("Name: MAL")
elif len(stats_filtradas) > 25:
    print("Name: MAL")
else:
    print("Name: OK")


#-----------------------Seguimos con DATE ---------------------------------------------------

# Binarizar el crop del campo Date: píxeles oscuros (texto) se vuelven 1
campo_date_th = (campo_date < 200).astype(np.uint8)

# Detectar componentes conectadas (grupos de píxeles conectados = caracteres)
# Devuelve: cantidad de componentes, imagen etiquetada, estadísticas (x,y,w,h,area), centroides
num_labels, labels, stats_date, centroids = cv2.connectedComponentsWithStats(campo_date_th, 8, cv2.CV_32S)

# Filtrar componentes por área: eliminar ruido (<10) y fondo/línea subrayada (>50)
stats_date_filtradas = stats_date[(stats_date[:, 4] > 10) & (stats_date[:, 4] < 50)]
print(f"Caracteres detectados: {len(stats_date_filtradas)}")

# Calcular gaps entre posiciones x consecutivas de cada carácter
gaps_date = np.diff(stats_date_filtradas[:, 0])

# Contar cuántos gaps superan 15px → indica espacios entre palabras
n_espacios_date = np.sum(gaps_date > 15)


if len(stats_date_filtradas) == 0:       
    print("Date: MAL")
elif len(stats_date_filtradas) != 8:     
    print("Date: MAL")
elif n_espacios_date > 0:                
    print("Date: MAL")
else:
    print("Date: OK")

#------------------------------Class------------------------------
#binarizamos la imagen y transformamos en uint8 para el connectedComoponentwithstats
campo_class_th = (campo_class < 200).astype(np.uint8)

#obtenemos las componentes
num_labels, labels, stats_class, centroids = cv2.connectedComponentsWithStats(campo_class_th, 8, cv2.CV_32S)

print(stats_class)

# filtramos por el area
stats_class_filtradas = stats_class[(stats_class[:, 4] > 10) & (stats_class[:, 4] < 100)]
print(f"Caracteres detectados: {len(stats_class_filtradas)}")

if len(stats_class_filtradas) == 0:
    print("Class: MAL")
elif len(stats_class_filtradas) != 1:
    print("Class: MAL")
else:
    print("Class: OK")


#------------------------celdas con preguntas ----------------------------------

celda1_th = (celdas[1] < 200).astype(np.uint8)
num_labels, labels, stats_c1, centroids = cv2.connectedComponentsWithStats(celda1_th, 8, cv2.CV_32S)
print(f"num_labels: {num_labels}")
print(stats_c1)

# Representacion de valores: x, y, w, h, area por cada fila (componente) Explicacion de porque 50 y 200?
th_area_min = 50
th_area_max = 200
stats_c1_filtradas = stats_c1[(stats_c1[:, 4] > th_area_min) & (stats_c1[:, 4] < th_area_max)]
print(f"Componentes filtradas: {len(stats_c1_filtradas)}")
print(stats_c1_filtradas)

# Encontrar el guión: ancho > 50 y alto == 1
guion = stats_c1_filtradas[(stats_c1_filtradas[:, 2] > 50) & (stats_c1_filtradas[:, 3] <= 2)]
print(f"Guión encontrado en: {guion}")

#Guion --------> x=91, y=36, w=114, h=1, area=114
y_guion = guion[0, 1]
zona_respuesta = celdas[1][:y_guion, :]

plt.figure(figsize=(8, 3))
plt.imshow(zona_respuesta, cmap='gray')
plt.title('Zona de respuesta')
plt.axis('off')
plt.show()

zona_th = (zona_respuesta < 200).astype(np.uint8)
num_labels, labels, stats_zona, centroids = cv2.connectedComponentsWithStats(zona_th, 8, cv2.CV_32S)
print(stats_zona)

stats_zona_f = stats_zona[(stats_zona[:, 4] > 20) & (stats_zona[:, 4] < 50)]
for comp in stats_zona_f:
    x, y, w, h, area = comp
    crop = zona_respuesta[y:y+h, x:x+w]
    plt.figure(figsize=(2, 2))
    plt.imshow(crop, cmap='gray')
    plt.title(f'x={x} y={y} area={area}')
    plt.axis('off')
    plt.show()

# x=139 y=23 area=35 --> B respuesta.
 

x_guion = guion[0, 0]
y_guion = guion[0, 1]
w_guion = guion[0, 2]

zona_guion = zona_respuesta[y_guion-15:y_guion, x_guion:x_guion + w_guion]

plt.figure(figsize=(4, 3))
plt.imshow(zona_guion, cmap='gray')
plt.title('Zona del guión')
plt.axis('off')
plt.show()

zona_guion_th = (zona_guion < 200).astype(np.uint8)
contours, hierarchy = cv2.findContours(zona_guion_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(f"Cantidad de contornos: {len(contours)}")
print(f"Jerarquía: {hierarchy}")
#jerarquia de contornos: [next, previous, first_child, parent]

# Contar contornos que tienen padre (son huecos interiores)
n_huecos = sum(1 for h in hierarchy[0] if h[3] != -1)
print(f"Huecos interiores: {n_huecos}")

if n_huecos == 2:
    letra_detectada = 'B'
elif n_huecos == 1:
    # contorno externo es el que no tiene padre
    contorno_externo = [i for i, h in enumerate(hierarchy[0]) if h[3] == -1][0]
    x, y, w, h = cv2.boundingRect(contours[contorno_externo])
    if h > w:
        letra_detectada = 'A'
    else:
        letra_detectada = 'D'
elif n_huecos == 0:
    letra_detectada = 'C'
print(f"Letra detectada: {letra_detectada}")

#CELDA 4
celda4_th = (celdas[4] < 200).astype(np.uint8)
_, _, stats_c4, _ = cv2.connectedComponentsWithStats(celda4_th, 8, cv2.CV_32S)
stats_c4_filtradas = stats_c4[(stats_c4[:, 4] > 50) & (stats_c4[:, 4] < 200)]
guion4 = stats_c4_filtradas[(stats_c4_filtradas[:, 2] > 50) & (stats_c4_filtradas[:, 3] <= 2)]
print(f"Guión: {guion4}")

x_g, y_g, w_g = guion4[0,0], guion4[0,1], guion4[0,2]
zona4 = celdas[4][y_g-15:y_g, x_g:x_g+w_g]

plt.figure(figsize=(4,3))
plt.imshow(zona4, cmap='gray')
plt.title('Zona D - celda 4')
plt.axis('off')
plt.show()

zona4_th = (zona4 < 200).astype(np.uint8)
contours4, hierarchy4 = cv2.findContours(zona4_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contorno_externo = [i for i, h in enumerate(hierarchy4[0]) if h[3] == -1][0]
x, y, w, h = cv2.boundingRect(contours4[contorno_externo])
print(f"w={w}, h={h}")



#Celda 8
celda8_th = (celdas[8] < 200).astype(np.uint8)
_, _, stats_c8, _ = cv2.connectedComponentsWithStats(celda8_th, 8, cv2.CV_32S)
stats_c8_filtradas = stats_c8[(stats_c8[:, 4] > 50) & (stats_c8[:, 4] < 200)]
guion8 = stats_c8_filtradas[(stats_c8_filtradas[:, 2] > 50) & (stats_c8_filtradas[:, 3] <= 2)]
print(f"Guión: {guion8}")

x_g, y_g, w_g = guion8[0,0], guion8[0,1], guion8[0,2]
zona8 = celdas[8][y_g-15:y_g, x_g:x_g+w_g]

plt.figure(figsize=(4,3))
plt.imshow(zona8, cmap='gray')
plt.title('Zona celda 8 - debería estar vacía')
plt.axis('off')
plt.show()

zona8_th = (zona8 < 200).astype(np.uint8)
print(f"Píxeles activos: {np.sum(zona8_th)}")



#Celda 6
celda6_th = (celdas[6] < 200).astype(np.uint8)
_, _, stats_c6, _ = cv2.connectedComponentsWithStats(celda6_th, 8, cv2.CV_32S)
stats_c6_filtradas = stats_c6[(stats_c6[:, 4] > 50) & (stats_c6[:, 4] < 200)]
guion6 = stats_c6_filtradas[(stats_c6_filtradas[:, 2] > 50) & (stats_c6_filtradas[:, 3] <= 2)]
x_g, y_g, w_g = guion6[0,0], guion6[0,1], guion6[0,2]
zona6 = celdas[6][y_g-15:y_g, x_g:x_g+w_g]
zona6_th = (zona6 < 200).astype(np.uint8)
contours6, hierarchy6 = cv2.findContours(zona6_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(f"Jerarquía: {hierarchy6}")
n_externos = sum(1 for h in hierarchy6[0] if h[3] == -1)
print(f"n_externos: {n_externos}")

#Celda 2
celda2_th = (celdas[2] < 200).astype(np.uint8)
_, _, stats_c2, _ = cv2.connectedComponentsWithStats(celda2_th, 8, cv2.CV_32S)
stats_c2_filtradas = stats_c2[(stats_c2[:, 4] > 50) & (stats_c2[:, 4] < 200)]
guion2 = stats_c2_filtradas[(stats_c2_filtradas[:, 2] > 50) & (stats_c2_filtradas[:, 3] <= 2)]
x_g, y_g, w_g = guion2[0,0], guion2[0,1], guion2[0,2]
zona2 = celdas[2][y_g-15:y_g, x_g:x_g+w_g]

plt.figure(figsize=(4,3))
plt.imshow(zona2, cmap='gray')
plt.title('Zona celda 2 - B y C')
plt.axis('off')
plt.show()

zona2_th = (zona2 < 200).astype(np.uint8)
contours2, hierarchy2 = cv2.findContours(zona2_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(f"Jerarquía: {hierarchy2}")
for i, (contour, h) in enumerate(zip(contours2, hierarchy2[0])):
    if h[3] == -1:
        area = cv2.contourArea(contour)
        print(f"Externo {i}: area={area}")


cols_zona = np.sum(zona_th, axis=0)

plt.figure(figsize=(8, 3))
plt.plot(cols_zona)
plt.grid(True)
plt.title('Proyección columnas zona celda 2')
plt.show()

for i, (contour, h) in enumerate(zip(contours2, hierarchy2[0])):
    area = cv2.contourArea(contour)
    print(f"Contorno {i}: area={area:.1f} padre={h[3]}")

x, y, w, h = cv2.boundingRect(contours2[1])  # externo grande celda 2
print(f"Celda 2 - w={w}")

x, y, w, h = cv2.boundingRect(contours[0])   # externo celda 1
print(f"Celda 1 - w={w}")

print(f"Celda 2 - píxeles activos: {np.sum(zona2_th)}")
print(f"Celda 1 - píxeles activos: {np.sum(zona_th)}")

#exploracion examen 1
#Exploracion examen 1
img1 = cv2.imread('TP_1/examen_1.png', cv2.IMREAD_GRAYSCALE)    
filas1, columnas1 = detectar_lineas(img1)
print(f"filas: {filas1} → {len(filas1)}")
print(f"columnas: {columnas1} → {len(columnas1)}")

fig, ax = plt.subplots(figsize=(8, 10))
ax.imshow(img1, cmap='gray')
for y in filas1:
    ax.axhline(y, color='red', linewidth=0.5)
for x in columnas1:
    ax.axvline(x, color='blue', linewidth=0.5)
plt.title('Líneas detectadas examen 1')
plt.show()

img1_th = (img1 < 200).astype(np.uint8)
img1_cols = np.sum(img1_th, axis=0)
raw_v1 = np.where(img1_cols > 400)[0]
print(f"raw_v1: {raw_v1}")

columnas = agrupar_lineas(raw_v1, gap=5).astype(int)

img1 = cv2.imread('TP_1/examen_1.png', cv2.IMREAD_GRAYSCALE)
filas1, columnas1 = detectar_lineas(img1)
_, enc1 = extraer_celdas(img1, filas1, columnas1)
campo_date1 = enc1[:, 279:354]
campo_date1_th = (campo_date1 < 200).astype(np.uint8)
_, _, stats_date1, _ = cv2.connectedComponentsWithStats(campo_date1_th, 8, cv2.CV_32S)
stats_date1_f = stats_date1[(stats_date1[:, 4] > 10) & (stats_date1[:, 4] < 50)]
gaps1 = np.diff(stats_date1_f[:, 0])
print(f"Caracteres: {len(stats_date1_f)}")
print(f"gaps: {gaps1}")

print(stats_date1_f)

campo_class1 = enc1[:, 404:530]
campo_class1_th = (campo_class1 < 200).astype(np.uint8)
_, _, stats_class1, _ = cv2.connectedComponentsWithStats(campo_class1_th, 8, cv2.CV_32S)
print(stats_class1)

print(stats_class1)

plt.figure(figsize=(3, 2))
plt.imshow(campo_class1, cmap='gray')
plt.title('Class examen 1')
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 2))
plt.imshow(enc1, cmap='gray')
plt.title('Encabezado examen 1')
plt.axis('off')
plt.show()
print(f"Ancho encabezado: {enc1.shape[1]}")

enc1_th = (enc1 < 200).astype(np.uint8)
enc1_cols = np.sum(enc1_th, axis=0)

plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1); plt.imshow(enc1, cmap='gray'); plt.axis('off')
plt.subplot(1, 2, 2); plt.plot(enc1_cols); plt.grid(True)
plt.title('Proyección columnas encabezado 1')
plt.tight_layout()
plt.show()

campo_name1 = enc1[:, int(enc1.shape[1]*0.08):int(enc1.shape[1]*0.43)]
campo_name1_th = (campo_name1 < 200).astype(np.uint8)
_, _, stats_name1, _ = cv2.connectedComponentsWithStats(campo_name1_th, 8, cv2.CV_32S)
stats_name1_f = stats_name1[(stats_name1[:, 4] > 20) & (stats_name1[:, 4] < 100)]
gaps_name1 = np.diff(stats_name1_f[:, 0])
print(f"gaps: {gaps_name1}")
print(f"max gap: {gaps_name1.max()}")

campo_name1 = enc1[:, 42:217]
campo_name1_th = (campo_name1 < 200).astype(np.uint8)
_, _, stats_name1, _ = cv2.connectedComponentsWithStats(campo_name1_th, 8, cv2.CV_32S)
stats_name1_f = stats_name1[(stats_name1[:, 4] > 20) & (stats_name1[:, 4] < 100)]
print(f"Componentes: {len(stats_name1_f)}")
print(stats_name1_f)
gaps_name1 = np.diff(stats_name1_f[:, 0])
print(f"gaps: {gaps_name1}")
print(f"n_palabras: {np.sum(gaps_name1 > 15) + 1}")

plt.figure(figsize=(6, 2))
plt.imshow(campo_name1, cmap='gray')
plt.title('Campo name examen 1')
plt.axis('off')
plt.show()

enc1_th = (enc1 < 200).astype(np.uint8)
enc1_cols = np.sum(enc1_th, axis=0)

plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1); plt.imshow(enc1, cmap='gray'); plt.axis('off')
plt.subplot(1, 2, 2); plt.plot(enc1_cols); plt.grid(True)
plt.axvline(42, color='red', linestyle='--', label='inicio actual')
plt.axvline(233, color='blue', linestyle='--', label='fin actual')
plt.legend()
plt.title('Proyección columnas encabezado 1')
plt.tight_layout()
plt.show()


#Exploracion examen 5
img5 = cv2.imread('TP_1/examen_5.png', cv2.IMREAD_GRAYSCALE)
filas5, columnas5 = detectar_lineas(img5)
_, enc5 = extraer_celdas(img5, filas5, columnas5)

plt.figure(figsize=(10, 2))
plt.imshow(enc5, cmap='gray')
plt.title('Encabezado examen 5')
plt.axis('off')
plt.show()

campo_date5 = enc5[:, 279:354]
campo_date5_th = (campo_date5 < 200).astype(np.uint8)
_, _, stats_date5, _ = cv2.connectedComponentsWithStats(campo_date5_th, 8, cv2.CV_32S)
print(stats_date5)

stats_date5_filtradas = stats_date5[(stats_date5[:, 4] > 10) & (stats_date5[:, 4] < 50)]
print(f"Caracteres detectados: {len(stats_date5_filtradas)}")
print(stats_date5_filtradas)

gaps_date5 = np.diff(stats_date5_filtradas[:, 0])
print(f"gaps: {gaps_date5}")
print(f"n_espacios: {np.sum(gaps_date5 > 15)}")

for i, path in enumerate(['TP_1/examen_1.png', 'TP_1/examen_2.png', 'TP_1/examen_3.png', 
                           'TP_1/examen_4.png', 'TP_1/examen_5.png']):
    img_tmp = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    filas_tmp, columnas_tmp = detectar_lineas(img_tmp)
    _, enc_tmp = extraer_celdas(img_tmp, filas_tmp, columnas_tmp)
    print(f"Examen {i+1}: ancho encabezado = {enc_tmp.shape[1]}")

#examen 2 
img2 = cv2.imread('TP_1/examen_2.png', cv2.IMREAD_GRAYSCALE)
filas2, columnas2 = detectar_lineas(img2)
celdas2, _ = extraer_celdas(img2, filas2, columnas2)
print(detectar_letra(celdas2[2]))

celda2_th = (celdas2[2] < 200).astype(np.uint8)
_, _, stats_c2, _ = cv2.connectedComponentsWithStats(celda2_th, 8, cv2.CV_32S)
stats_c2_f = stats_c2[(stats_c2[:, 4] > 50) & (stats_c2[:, 4] < 200)]
guion2 = stats_c2_f[(stats_c2_f[:, 2] > 50) & (stats_c2_f[:, 3] <= 2)]
x_g, y_g, w_g = guion2[0,0], guion2[0,1], guion2[0,2]
zona2 = celdas2[2][y_g-15:y_g, x_g:x_g+w_g]

plt.figure(figsize=(4,3))
plt.imshow(zona2, cmap='gray')
plt.title('Zona celda 2')
plt.axis('off')
plt.show()

zona2_th = (zona2 < 200).astype(np.uint8)
contours2, hierarchy2 = cv2.findContours(zona2_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for i, (c, h) in enumerate(zip(contours2, hierarchy2[0])):
    print(f"Contorno {i}: area={cv2.contourArea(c):.1f} padre={h[3]}")


img = cv2.imread('TP_1/examen_3.png', cv2.IMREAD_GRAYSCALE)
filas, columnas = detectar_lineas(img)
celdas, encabezado = extraer_celdas(img, filas, columnas)

# Inspeccionamos celda 6 y 8 examen 3, dan mal pero deberian dar bien
img = cv2.imread('TP_1/examen_2.png', cv2.IMREAD_GRAYSCALE)
filas, columnas = detectar_lineas(img)
celdas, encabezado = extraer_celdas(img, filas, columnas)

celda = celdas[6]
celda_th = (celda < 200).astype(np.uint8)

_, _, stats, _ = cv2.connectedComponentsWithStats(celda_th, 8, cv2.CV_32S)
stats_filtradas = stats[(stats[:, 4] > 50) & (stats[:, 4] < 200)]
print("stats_filtradas:\n", stats_filtradas)

guion = stats_filtradas[(stats_filtradas[:, 2] > 50) & (stats_filtradas[:, 3] <= 2)]
print("guion:\n", guion)

x_guion = guion[0, 0]
y_guion = guion[0, 1]
w_guion = guion[0, 2]

print(f"x={x_guion}, y={y_guion}, w={w_guion}")
print(f"zona: [{y_guion-15}:{y_guion}]")

zona = celda[y_guion-14:y_guion, x_guion:x_guion + w_guion]
zona_th = (zona < 200).astype(np.uint8)

plt.imshow(zona, cmap='gray')
plt.title('Zona')
plt.axis('off')
plt.show()

print(f"pixeles activos: {np.sum(zona_th)}")

zona_th = (zona < 200).astype(np.uint8)
_, _, stats_zona, _ = cv2.connectedComponentsWithStats(zona_th, 8, cv2.CV_32S)
print(stats_zona)