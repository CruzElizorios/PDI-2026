import cv2
import numpy as np
import matplotlib.pyplot as plt

RESPUESTAS_CORRECTAS = {1:'C', 2:'B', 3:'A', 4:'D', 5:'B', 6:'B', 7:'A', 8:'B', 9:'D', 10:'D'}

def agrupar_lineas(indices, gap=10):
    if len(indices) == 0:
        return np.array([])
    grupos = []
    inicio = indices[0]
    anterior = indices[0]
    for i in indices[1:]:
        if i - anterior > gap:
            grupos.append((inicio + anterior) // 2)
            inicio = i
        anterior = i
    grupos.append((inicio + anterior) // 2)
    return np.array(grupos)

def detectar_lineas(img):
    img_th = (img < 200).astype(np.uint8)
    img_rows = np.sum(img_th, axis=1)
    img_cols = np.sum(img_th, axis=0)
    raw_h = np.where(img_rows > 290)[0]
    raw_v = np.where(img_cols > 400)[0]
    filas    = agrupar_lineas(raw_h, gap=10).astype(int)
    #columnas = agrupar_lineas(raw_v, gap=30).astype(int)
    columnas = agrupar_lineas(raw_v, gap=5).astype(int)
    return filas, columnas

def extraer_celdas(img, filas, columnas):
    celdas = {}
    rango_filas = [(filas[i+1], filas[i+2]) for i in range(5)]
    rango_cols  = [(columnas[0], columnas[1]), (columnas[2], columnas[3])]
    for i, (y1, y2) in enumerate(rango_filas):
        for j, (x1, x2) in enumerate(rango_cols):
            nro = i + 1 if j == 0 else i + 6
            celdas[nro] = img[y1:y2, x1:x2]
    encabezado = img[0:filas[1], columnas[0]:columnas[3]]
    return celdas, encabezado

def detectar_campos(encabezado):
    enc_th = (encabezado < 200).astype(np.uint8)
    enc_rows = np.sum(enc_th, axis=1)
    fila_linea = np.where(enc_rows > 200)[0][0]
    fila = enc_th[fila_linea, :]
    diff_fila = np.diff(fila, prepend=0, append=0)
    inicios_campos = np.where(diff_fila == 1)[0]
    fines_campos   = np.where(diff_fila == -1)[0]
    return inicios_campos, fines_campos

def validar_encabezado(encabezado):
    # Crops de cada campo
    inicios_campos, fines_campos = detectar_campos(encabezado)

    campo_name  = encabezado[:, inicios_campos[0]:fines_campos[0]]
    campo_date  = encabezado[:, inicios_campos[1]:fines_campos[1]]
    campo_class = encabezado[:, inicios_campos[2]:fines_campos[2]]

    # --- Name ---
    campo_name_th = (campo_name < 200).astype(np.uint8)
    _, _, stats_name, _ = cv2.connectedComponentsWithStats(campo_name_th, 8, cv2.CV_32S)
    stats_name = stats_name[(stats_name[:, 4] > 20) & (stats_name[:, 4] < 100)]
    if len(stats_name) == 0:
        name_ok = False
    else:
        gaps = np.diff(stats_name[:, 0])
        n_palabras = np.sum(gaps > 15) + 1
        name_ok = n_palabras == 2 and len(stats_name) <= 25

    # --- Date ---
    campo_date_th = (campo_date < 200).astype(np.uint8)
    _, _, stats_date, _ = cv2.connectedComponentsWithStats(campo_date_th, 8, cv2.CV_32S)
    stats_date = stats_date[(stats_date[:, 4] > 15) & (stats_date[:, 4] < 50)]
    if len(stats_date) == 0:
        date_ok = False
    else:
        gaps_date = np.diff(stats_date[:, 0])
        date_ok = len(stats_date) == 8 and np.sum(gaps_date > 25) == 0


    # --- Class ---
    campo_class_th = (campo_class < 200).astype(np.uint8)
    _, _, stats_class, _ = cv2.connectedComponentsWithStats(campo_class_th, 8, cv2.CV_32S)
    stats_class = stats_class[(stats_class[:, 4] > 10) & (stats_class[:, 4] < 100)]
    class_ok = len(stats_class) == 1

    print(f"Name:  {'OK' if name_ok  else 'MAL'}")
    print(f"Date:  {'OK' if date_ok  else 'MAL'}")
    print(f"Class: {'OK' if class_ok else 'MAL'}")

def detectar_letra(celda):
    celda_th = (celda < 200).astype(np.uint8)

    # Buscar el guión
    _, _, stats, _ = cv2.connectedComponentsWithStats(celda_th, 8, cv2.CV_32S)
    stats_filtradas = stats[(stats[:, 4] > 50) & (stats[:, 4] < 200)]
    guion = stats_filtradas[(stats_filtradas[:, 2] > 50) & (stats_filtradas[:, 3] <= 2)]

    if len(guion) == 0:
        return 'MAL'

    x_guion = guion[0, 0]
    y_guion = guion[0, 1]
    w_guion = guion[0, 2]

    # Crop zona de respuesta
    zona = celda[y_guion-15:y_guion, x_guion:x_guion + w_guion]
    zona_th = (zona < 200).astype(np.uint8)

    # Si hay muy pocos píxeles activos → zona vacía
    if np.sum(zona_th) < 20:
        return 'MAL'
    
    # Detectar contornos
    contours, hierarchy = cv2.findContours(zona_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None or len(contours) == 0:
        return 'MAL'
    
    # Si hay más de 3 contornos → más de una letra
    if len(contours) > 3:
        return 'MAL - Múltiples contornos detectados'
    
    # Contar contornos externos con área suficiente (ignorar ruido pequeño)
    n_externos = sum(1 for i, h in enumerate(hierarchy[0]) 
                    if h[3] == -1 and cv2.contourArea(contours[i]) > 20)

    if n_externos > 1:
        return 'MAL'
    
    n_huecos = sum(1 for i, h in enumerate(hierarchy[0]) 
                if h[3] != -1 and cv2.contourArea(contours[i]) > 5)

    if n_huecos > 2:
        return 'MAL'

    if n_huecos == 2:
        return 'B'
    elif n_huecos == 0:
        return 'C'
    elif n_huecos == 1:       
        # # Obtenemos el contorno externo para aislar la letra
        contorno_externo_idx = [i for i, h in enumerate(hierarchy[0]) if h[3] == -1][0]
        x_ext, y_ext, w_ext, h_ext = cv2.boundingRect(contours[contorno_externo_idx])
        
        # Recortamos la letra de la zona binarizada para un análisis limpio
        img_letra = zona_th[y_ext:y_ext+h_ext, x_ext:x_ext+w_ext]
        
        # Calculamos el perfil de proyección vertical (Unidad 1 y 3)
        # Sumamos los píxeles blancos (valor 1) por cada columna.
        suma_vertical_columnas = np.sum(img_letra, axis=0)
        
        # se busca la línea recta:
        # Si alguna columna tiene una suma de píxeles mayor al 80% de la altura total
        # de la letra, asumimos que existe una línea vertical (característica de la 'D').
        umbral_linea_recta = 0.8 * h_ext
        tiene_linea_vertical = np.any(suma_vertical_columnas > umbral_linea_recta)

        if tiene_linea_vertical:
            return 'D'
        else:
            return 'A'

        # # # --- Método alternativo para diferenciar A y D (si hay un solo hueco) ---
        # # Obtener bounding box del contorno externo
        # contorno_externo_idx = [i for i, h in enumerate(hierarchy[0]) if h[3] == -1][0]
        # x_ext, y_ext, w_ext, h_ext = cv2.boundingRect(contours[contorno_externo_idx])
        
        # # Obtener centroide del hueco (contorno con padre)
        # contorno_hueco_idx = [i for i, h in enumerate(hierarchy[0]) if h[3] != -1][0]
        # x_h, y_h, w_h, h_h = cv2.boundingRect(contours[contorno_hueco_idx])
        
        # # Centro del hueco relativo al contorno externo
        # centro_hueco = y_h + h_h/2
        # centro_letra = y_ext + h_ext/2
        
        # # Si el hueco está en la mitad superior → A, sino → D
        # if centro_hueco < centro_letra:
        #     return 'A'
        # else:
        #     return 'D'
    else:
        return 'MAL'

def corregir_examen(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    filas, columnas = detectar_lineas(img)
    celdas, encabezado = extraer_celdas(img, filas, columnas)

    print(f"\n--- Examen: {img_path} ---")
    validar_encabezado(encabezado)

    correctas = 0
    for nro in range(1, 11):
        letra = detectar_letra(celdas[nro])
        correcta = RESPUESTAS_CORRECTAS[nro]
        if letra == correcta:
            resultado = 'OK'
            correctas += 1
        else:
            resultado = 'MAL'
        print(f"Pregunta {nro:2d}: {resultado}  (detectada: {letra}, correcta: {correcta})")

    aprobado = correctas >= 6
    print(f"Resultado: {'APROBADO' if aprobado else 'DESAPROBADO'} ({correctas}/10 correctas)")
    return aprobado, encabezado

# --- Ejecutar sobre todos los exámenes ---
if __name__ == '__main__':
    examenes = ['TP_1/examen_1.png', 'TP_1/examen_2.png', 'TP_1/examen_3.png',
                'TP_1/examen_4.png', 'TP_1/examen_5.png']

    resultados = []
    for path in examenes:
        aprobado, encabezado = corregir_examen(path)
        inicios_campos, fines_campos = detectar_campos(encabezado)
        name_crop = encabezado[:, inicios_campos[0]:fines_campos[0]]
        resultados.append((name_crop, aprobado))


    fig, axes = plt.subplots(5, 1, figsize=(8, 10))
    for i, (crop, aprobado) in enumerate(resultados):
        axes[i].imshow(crop, cmap='gray')
        axes[i].set_title('APROBADO' if aprobado else 'DESAPROBADO', 
                        color='green' if aprobado else 'red')
        axes[i].axis('off')
    plt.show()


pruebas = ['tp1/examen_1.png',
           'tp1/examen_2.png',
           'tp1/examen_3.png',
           'tp1/examen_4.png',
           'tp1/examen_5.png'
          ]
for path in pruebas:
    corregir_examen(path)