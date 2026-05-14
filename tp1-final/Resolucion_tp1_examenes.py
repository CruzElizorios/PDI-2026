import cv2
import numpy as np
import matplotlib.pyplot as plt

RESPUESTAS_CORRECTAS = {1:'C', 2:'B', 3:'A', 4:'D', 5:'B', 6:'B', 7:'A', 8:'B', 9:'D', 10:'D'}

def agrupar_lineas(indices, gap=10):
    """
    Colapsa runs de índices consecutivos en un único centroide por línea.

    Las líneas impresas tienen grosor de varios píxeles, por lo que la
    proyección de intensidades (con np.where) genera clusters de índices en lugar de
    un pico único. Se fusionan índices cuya separación no supere `gap`.

    Args:
        indices (np.ndarray): Índices de filas/columnas con alta densidad
            de píxeles oscuros.
        gap (int): Separación máxima para considerar dos índices como parte
            de la misma línea.

    Returns:
        np.ndarray: Un índice central por línea detectada.
    """
    if len(indices) == 0:
        return np.array([])
    grupos = []
    inicio = indices[0] # marca dónde empieza el run actual
    anterior = indices[0] # puntero al último índice visto, avanza en cada iteración
    for i in indices[1:]:
        if i - anterior > gap:
            # salto grande → el run terminó
            # el centroide es el promedio entre donde empezó y donde terminó
            grupos.append((inicio + anterior) // 2)
            inicio = i
        anterior = i
    grupos.append((inicio + anterior) // 2) # se agrega el ultimo run de manera manual
    return np.array(grupos)


def detectar_lineas(img):
    """
    Localiza las líneas de grilla del formulario por proyección de intensidades.

    Binarizar y sumar píxeles activos por fila/columna permite detectar
    líneas estructurales.
    Las filas/columnas cuya suma supera un umbral empírico contienen una línea.

    Args:
        img (np.ndarray): Imagen en escala de grises del formulario.

    Returns:
        tuple[np.ndarray, np.ndarray]: Índices de líneas horizontales y
            verticales de la grilla.
    """
    img_th = (img < 200).astype(np.uint8)
    img_rows = np.sum(img_th, axis=1) #proyeccion horizontal 
    img_cols = np.sum(img_th, axis=0) #proyeccion vertical
    raw_h = np.where(img_rows > 290)[0] # Filas/columnas que superan el umbral → contienen una línea de grilla
    raw_v = np.where(img_cols > 400)[0] # (umbral empírico: una línea real activa muchos más píxeles que texto o ruido)
    filas    = agrupar_lineas(raw_h, gap=10).astype(int) # Colapsar runs de índices consecutivos en un único centroide por línea
    columnas = agrupar_lineas(raw_v, gap=5).astype(int)
    return filas, columnas


def extraer_celdas(img, filas, columnas):
    """
    Segmenta el formulario en ROIs de cada celda de respuesta y el encabezado.

    Con las líneas de grilla como delimitadores espaciales, se indexa
    directamente sobre la imagen para extraer cada región sin transformaciones
    geométricas. Las 10 celdas se distribuyen en 2 columnas de 5 filas.

    Args:
        img (np.ndarray): Imagen en escala de grises del formulario.
        filas (np.ndarray): Índices de líneas horizontales.
        columnas (np.ndarray): Índices de líneas verticales.

    Returns:
        tuple[dict, np.ndarray]: Diccionario {nro_pregunta: ROI} y ROI
            del encabezado (nombre, fecha, comisión).
    """
    celdas = {}
    rango_filas = [(filas[i+1], filas[i+2]) for i in range(5)]
    rango_cols  = [(columnas[0], columnas[1]), (columnas[2], columnas[3])]
    for i, (y1, y2) in enumerate(rango_filas):
        for j, (x1, x2) in enumerate(rango_cols):
            nro = i + 1 if j == 0 else i + 6 #asigno valor de celda (pregunta)
            celdas[nro] = img[y1:y2, x1:x2] # realizo crop
    encabezado = img[0:filas[1], columnas[0]:columnas[3]] 
    return celdas, encabezado


def detectar_campos(encabezado):
    """
    Detecta los límites horizontales de los campos del encabezado usando
    la línea base impresa como referencia.

    La línea impresa bajo cada campo es el elemento más denso del encabezado.
    Calcular la derivada discreta sobre esa fila binarizada produce flancos
    +1/-1 que marcan inicio y fin de cada segmento continuo de tinta.

    Args:
        encabezado (np.ndarray): ROI del encabezado en escala de grises.

    Returns:
        tuple[np.ndarray, np.ndarray]: Coordenadas x de inicio y fin de
            cada campo (nombre, fecha, comisión).
    """
    enc_th = (encabezado < 200).astype(np.uint8)
    enc_rows = np.sum(enc_th, axis=1) # Proyección horizontal: suma de píxeles activos por fila
    fila_linea = np.where(enc_rows > 200)[0][0] # fila mas densa (linea del campo)
    fila = enc_th[fila_linea, :] # extraigo fila binaria
    diff_fila = np.diff(fila, prepend=0, append=0) # Derivada discreta: detecta flancos +1 (0→1) y -1 (1→0) ->prepend y append acomodan flancos inicio y fin
    inicios_campos = np.where(diff_fila == 1)[0] 
    fines_campos   = np.where(diff_fila == -1)[0]
    return inicios_campos, fines_campos


def validar_encabezado(encabezado):
    """
    Verifica que nombre, fecha y comisión estén correctamente completados,
    usando análisis de componentes conectadas sobre cada campo.

    En lugar de OCR (reconocimiento optico de caracter), se explotan propiedades morfológicas específicas de
    cada campo: cantidad de blobs (digitos), gaps inter-palabra y geometría. Esto es
    suficiente para detectar campos vacíos o mal completados sin reconocer
    el texto exacto.

      - Nombre: 2 palabras (gaps > 15 px) y ≤ 25 componentes.
      - Fecha (DD/MM/AAAA): exactamente 8 componentes sin gaps grandes.
      - Comisión: exactamente 1 componente (un único carácter).

    Args:
        encabezado (np.ndarray): ROI del encabezado en escala de grises.
    """
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
    """
    Clasifica la respuesta manuscrita (A/B/C/D) por topología de contornos.

    Cada letra tiene una firma topológica única en términos de regiones
    cerradas internas (huecos), lo que permite clasificar sin descriptores
    de forma ni templates:
      - C → 0 huecos (trazo abierto)
      - A → 1 hueco en la mitad superior del bounding box externo
      - D → 1 hueco en la mitad inferior
      - B → 2 huecos

    El guión impreso en la celda actúa como ancla espacial para localizar
    el ROI de escritura exactamente encima de él, evitando procesar ruido
    del resto de la celda. Se retorna 'MAL' ante respuesta vacía, múltiple
    o topología ambigua.

    Args:
        celda (np.ndarray): ROI en escala de grises de una celda de respuesta.

    Returns:
        str: 'A', 'B', 'C', 'D', o 'MAL' si la respuesta es inválida.
    """
    celda_th = (celda < 200).astype(np.uint8)

    _, _, stats, _ = cv2.connectedComponentsWithStats(celda_th, 8, cv2.CV_32S)
    #Filtramos por area para deshacernos del fondo y del ruido (50-200 porque las letras de respuestas son las mas grandes)
    stats_filtradas = stats[(stats[:, 4] > 50) & (stats[:, 4] < 200)]
    #Identificamos el guion (el mas ancho y bajo)
    guion = stats_filtradas[(stats_filtradas[:, 2] > 50) & (stats_filtradas[:, 3] <= 2)]

    if len(guion) == 0:
        return 'MAL'

    #obtenemos coordenadas para hacer los crops de cada respuesta.
    x_guion = guion[0, 0]
    y_guion = guion[0, 1]
    w_guion = guion[0, 2]

    #crop quitamos del alto la primera linea que incluye texto de la pregunta.
    zona = celda[y_guion-14:y_guion, x_guion:x_guion + w_guion]
    zona_th = (zona < 200).astype(np.uint8)

    #zona vacia, sin trazo manuscrito.
    if np.sum(zona_th) < 20:
        return 'MAL'

    # RETR_TREE recupera la jerarquía completa: permite distinguir
    contours, hierarchy = cv2.findContours(zona_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None or len(contours) == 0:
        return 'MAL'

    #no hay letra que tenga mas de 3 contornos
    if len(contours) > 3:
        return 'MAL'

    # Más de un contorno externo con área suficiente → más de una letra escrita
    n_externos = sum(1 for i, h in enumerate(hierarchy[0])
                    if h[3] == -1 and cv2.contourArea(contours[i]) > 20)

    if n_externos > 1:
        return 'MAL'

    # Huecos: contornos internos (con padre) con área > 5 px
    # Su cantidad determina la letra: C=0, A o D=1, B=2
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


def corregir_examen(img_path):
    """
    Pipeline completo de corrección automática de un formulario escaneado.

    Orquesta detección de grilla → segmentación de ROIs → validación de
    encabezado → clasificación de respuestas → cálculo de resultado.
    El umbral de aprobación es 6/10 respuestas correctas.

    Args:
        img_path (str): Ruta a la imagen del formulario escaneado.

    Returns:
        tuple[bool, np.ndarray]: Estado de aprobación y ROI del encabezado
            (para visualización del nombre del alumno).
    """
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
    examenes = ['tp1-final/examen_1.png', 'tp1-final/examen_2.png', 'tp1-final/examen_3.png',
                'tp1-final/examen_4.png', 'tp1-final/examen_5.png']

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


