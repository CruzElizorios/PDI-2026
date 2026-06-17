from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ==============================================================================
# CONFIGURACIÓN GENERAL
# ==============================================================================

ESCALA_MAX = 800

# Para trabajar sobre una sola imagen y no abrir demasiadas ventanas:
NUMEROS_IMAGENES = [6]

# Para procesar las 12 imágenes, reemplazar la línea anterior por:
#NUMEROS_IMAGENES = range(1, 13)

MOSTRAR_FIGURAS = True
GUARDAR_FIGURAS = True
DPI_GUARDADO = 200

BASE_DIR = Path(__file__).resolve().parent
CARPETA_IMAGENES = BASE_DIR / "tp2-pruebas"
CARPETA_SALIDA = BASE_DIR / "figuras_informe"


# ==============================================================================
# PARÁMETROS DEL DETECTOR
# ==============================================================================

# Filtro geométrico de candidatos
RATIO_MIN = 2.0
RATIO_MAX = 4.5
AREA_REL_MIN = 0.005
AREA_REL_MAX = 0.05

# Scoring: relación de aspecto
RATIO_IDEAL = 3.08
UMBRAL_ANGULO = 15
DIST_RATIO_MAX = 0.5

# Scoring: presencia de píxeles blancos y negros
UMBRAL_BLANCO = 200
UMBRAL_NEGRO = 100
UMBRAL_SOLAP_BLANCO = 0.35
UMBRAL_SOLAP_NEGRO = 0.10

# Scoring: contornos con forma de carácter
RATIO_CAR_MIN = 1.2
RATIO_CAR_MAX = 2.5
AREA_CAR_MIN = 0.04
AREA_CAR_MAX = 0.20
N_CAR_MIN = 3
N_CAR_MAX = 7

# Scoring: franja azul superior
UMBRAL_SOLAP_AZUL = 0.25
PROPORCION_FRANJA = 0.20
AZUL_HSV_BAJO = np.array([100, 80, 50])
AZUL_HSV_ALTO = np.array([130, 255, 255])

# Segmentación final de caracteres
RATIO_SEG_MIN = 1.0
RATIO_SEG_MAX = 4.0
AREA_SEG_MIN = 0.02
AREA_SEG_MAX = 0.25


# ==============================================================================
# UTILIDADES DE VISUALIZACIÓN
# ==============================================================================

def mostrar_en_eje(ax, imagen, titulo, cmap=None):
    """Muestra una imagen en un eje de Matplotlib sin ejes cartesianos."""
    ax.imshow(imagen, cmap=cmap)
    ax.set_title(titulo)
    ax.axis("off")


def guardar_figura(fig, nombre_archivo):
    """Guarda una figura en la carpeta configurada para el informe."""
    if not GUARDAR_FIGURAS:
        return

    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
    ruta_salida = CARPETA_SALIDA / nombre_archivo
    fig.savefig(ruta_salida, dpi=DPI_GUARDADO, bbox_inches="tight")
    print(f"  Figura guardada: {ruta_salida}")


def dibujar_candidatos(img_rgb, candidatos, color=(255, 255, 0)):
    """Dibuja todos los candidatos y su número de identificación."""
    salida = img_rgb.copy()

    for i, (x, y, w, h, ratio, area_rel, cnt) in enumerate(candidatos):
        cv2.rectangle(salida, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            salida,
            f"ID {i}",
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return salida


def dibujar_ganador(img_rgb, candidato, puntaje):
    """Dibuja la caja contenedora del candidato seleccionado."""
    salida = img_rgb.copy()
    x, y, w, h, ratio, area_rel, cnt = candidato

    cv2.rectangle(salida, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.putText(
        salida,
        f"Patente - {puntaje} puntos",
        (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return salida


# ==============================================================================
# 1. CARGA Y REDIMENSIONAMIENTO
# ==============================================================================

def cargar_imagen(path):
    """
    Lee la imagen, la redimensiona y genera las versiones BGR, RGB y gris.
    """
    img_bgr = cv2.imread(str(path))

    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer: {path}")

    h, w = img_bgr.shape[:2]
    escala = ESCALA_MAX / max(h, w)

    img_bgr = cv2.resize(
        img_bgr,
        (int(w * escala), int(h * escala)),
        interpolation=cv2.INTER_AREA,
    )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    print(f"  Shape redimensionada: {img_bgr.shape} (escala={escala:.3f})")

    return img_bgr, img_rgb, img_gray


# ==============================================================================
# 2. PIPELINE PRINCIPAL: SUAVIZADO, SHARPENING Y CANNY
# ==============================================================================

def preprocesar_principal(img_gray):
    """
    Ejecuta el mismo pipeline usado para detectar bordes y devuelve cada etapa.
    """
    img_suave_3 = cv2.GaussianBlur(img_gray, (3, 3), 0)

    kernel_sharp = np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ],
        dtype=np.float32,
    )

    img_sharp = cv2.filter2D(img_suave_3, -1, kernel_sharp)
    img_blur_5 = cv2.GaussianBlur(img_sharp, (5, 5), 0)
    canny = cv2.Canny(img_blur_5, 100, 200)

    etapas = {
        "gris": img_gray,
        "gaussiano_3x3": img_suave_3,
        "sharpening": img_sharp,
        "gaussiano_5x5": img_blur_5,
        "canny": canny,
    }

    return canny, etapas


# ==============================================================================
# 3. EXTRACCIÓN DE CANDIDATOS DEL PIPELINE PRINCIPAL
# ==============================================================================

def extraer_candidatos(edges, img_area):
    """Filtra los contornos de Canny por relación de aspecto y área relativa."""
    contornos, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    print(f"  Total contornos encontrados con Canny: {len(contornos)}")

    candidatos = []

    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)

        if h == 0:
            continue

        ratio = w / h
        area_rel = (w * h) / img_area

        if RATIO_MIN < ratio < RATIO_MAX and AREA_REL_MIN < area_rel < AREA_REL_MAX:
            candidatos.append((x, y, w, h, ratio, area_rel, cnt))

    candidatos = sorted(candidatos, key=lambda c: c[5], reverse=True)
    print(f"  Candidatos principales: {len(candidatos)}")

    return candidatos


# ==============================================================================
# 4. PIPELINE ALTERNATIVO: BLACKHAT, SOBEL, OTSU Y MORFOLOGÍA
# ==============================================================================

def extraer_candidatos_alternativo(img_gray, img_area):
    """
    Ejecuta el detector alternativo y devuelve tanto los candidatos como todas
    las imágenes intermedias necesarias para documentar el procedimiento.
    """
    kernel_blackhat = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel_blackhat)

    sobel_float = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
    sobel_absoluto = np.absolute(sobel_float)

    sobel_normalizado = cv2.normalize(
        sobel_absoluto,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    ).astype("uint8")

    sobel_suavizado = cv2.GaussianBlur(sobel_normalizado, (5, 5), 0)

    umbral_otsu, binaria_otsu = cv2.threshold(
        sobel_suavizado,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    cierre = cv2.morphologyEx(binaria_otsu, cv2.MORPH_CLOSE, kernel_close)
    dilatada = cv2.dilate(cierre, None, iterations=2)

    contornos, _ = cv2.findContours(
        dilatada,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidatos = []

    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)

        if h == 0:
            continue

        ratio = w / h
        area_rel = (w * h) / img_area

        if 2.0 < ratio < 6.0 and 0.003 < area_rel < 0.08:
            candidatos.append((x, y, w, h, ratio, area_rel, cnt))

    candidatos = sorted(candidatos, key=lambda c: c[5], reverse=True)

    etapas = {
        "blackhat": blackhat,
        "sobel_normalizado": sobel_normalizado,
        "sobel_suavizado": sobel_suavizado,
        "otsu": binaria_otsu,
        "cierre": cierre,
        "dilatada": dilatada,
        "umbral_otsu": umbral_otsu,
    }

    print(f"  Candidatos alternativos: {len(candidatos)}")
    print(f"  Umbral seleccionado por Otsu: {umbral_otsu:.1f}")

    return candidatos, etapas


# ==============================================================================
# 5. PUNTUACIÓN DE CANDIDATOS
# ==============================================================================

def puntuar_candidatos(candidatos, img_gray, img_bgr):
    """Aplica los cuatro criterios de puntuación del detector original."""
    puntajes = [0] * len(candidatos)

    # --------------------------------------------------------------------------
    # Criterio 1: relación de aspecto cercana al valor ideal
    # --------------------------------------------------------------------------
    for i, (x, y, w, h, ratio, area_rel, cnt) in enumerate(candidatos):
        rect = cv2.minAreaRect(cnt)
        _, (rw, rh), angulo = rect

        if abs(angulo) > UMBRAL_ANGULO and min(rw, rh) > 0:
            ratio_usado = max(rw, rh) / min(rw, rh)
        else:
            ratio_usado = ratio

        if abs(ratio_usado - RATIO_IDEAL) < DIST_RATIO_MAX:
            puntajes[i] += 1

    # --------------------------------------------------------------------------
    # Criterio 2: convivencia de fondo claro y caracteres oscuros
    # --------------------------------------------------------------------------
    mascara_blanco = (img_gray > UMBRAL_BLANCO).astype(np.uint8)
    mascara_negro = (img_gray < UMBRAL_NEGRO).astype(np.uint8)

    for i, (x, y, w, h, ratio, area_rel, cnt) in enumerate(candidatos):
        roi_blanco = mascara_blanco[y:y + h, x:x + w]
        roi_negro = mascara_negro[y:y + h, x:x + w]

        solap_blanco = roi_blanco.sum() / (w * h)
        solap_negro = roi_negro.sum() / (w * h)

        if solap_blanco > UMBRAL_SOLAP_BLANCO and solap_negro > UMBRAL_SOLAP_NEGRO:
            puntajes[i] += 1

    # --------------------------------------------------------------------------
    # Criterio 3: cantidad razonable de contornos con forma de carácter
    # --------------------------------------------------------------------------
    for i, (x, y, w, h, ratio, area_rel, cnt) in enumerate(candidatos):
        roi = img_gray[y:y + h, x:x + w]
        roi_area = w * h

        roi_bin = cv2.adaptiveThreshold(
            roi,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )

        contornos_roi, _ = cv2.findContours(
            roi_bin,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cantidad_caracteres = 0

        for c in contornos_roi:
            xc, yc, wc, hc = cv2.boundingRect(c)

            if hc == 0 or wc == 0:
                continue

            area_rel_c = (wc * hc) / roi_area
            ratio_c = hc / wc

            if (
                RATIO_CAR_MIN < ratio_c < RATIO_CAR_MAX
                and AREA_CAR_MIN < area_rel_c < AREA_CAR_MAX
            ):
                cantidad_caracteres += 1

        if N_CAR_MIN <= cantidad_caracteres <= N_CAR_MAX:
            puntajes[i] += 1

    # --------------------------------------------------------------------------
    # Criterio 4: presencia de azul en la franja superior
    # --------------------------------------------------------------------------
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mascara_azul = cv2.inRange(img_hsv, AZUL_HSV_BAJO, AZUL_HSV_ALTO)

    for i, (x, y, w, h, ratio, area_rel, cnt) in enumerate(candidatos):
        h_franja = max(1, int(h * PROPORCION_FRANJA))
        roi_azul = mascara_azul[y:y + h_franja, x:x + w]
        solap_azul = roi_azul.sum() / (255 * w * h_franja)

        if solap_azul > UMBRAL_SOLAP_AZUL:
            puntajes[i] += 1

    return puntajes, mascara_azul


def seleccionar_ganador(candidatos, puntajes):
    """Selecciona el mayor puntaje y desempata eligiendo el objeto más bajo."""
    puntaje_maximo = max(puntajes)
    indices = [i for i, p in enumerate(puntajes) if p == puntaje_maximo]

    if len(indices) == 1:
        return indices[0]

    return max(indices, key=lambda i: candidatos[i][1])


# ==============================================================================
# 6. SEGMENTACIÓN FINAL DE CARACTERES
# ==============================================================================

def segmentar_caracteres(crop_rgb, crop_gray):
    """
    Prueba distintos tamaños de vecindario y conserva el resultado cuya cantidad
    de componentes se aproxima más a los siete caracteres de la patente.
    """
    roi_h, roi_w = crop_gray.shape[:2]
    roi_area = roi_h * roi_w

    mejores_caracteres = []
    mejor_binaria = None
    mejor_block_size = None
    mejor_diferencia = float("inf")

    for block_size in [7, 9, 11, 13, 15]:
        binaria = cv2.adaptiveThreshold(
            crop_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            2,
        )

        contornos, _ = cv2.findContours(
            binaria,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        caracteres = []

        for c in contornos:
            x, y, w, h = cv2.boundingRect(c)

            if h == 0 or w == 0:
                continue

            area_rel = (w * h) / roi_area
            ratio = h / w

            if (
                RATIO_SEG_MIN < ratio < RATIO_SEG_MAX
                and AREA_SEG_MIN < area_rel < AREA_SEG_MAX
            ):
                caracteres.append((x, y, w, h))

        diferencia = abs(len(caracteres) - 7)

        if diferencia < mejor_diferencia:
            mejor_diferencia = diferencia
            mejores_caracteres = caracteres
            mejor_binaria = binaria
            mejor_block_size = block_size

    # Orden visual de izquierda a derecha
    mejores_caracteres = sorted(mejores_caracteres, key=lambda c: c[0])

    crop_debug = crop_rgb.copy()

    for x, y, w, h in mejores_caracteres:
        cv2.rectangle(crop_debug, (x, y), (x + w, y + h), (0, 255, 0), 1)

    print(
        f"  Caracteres válidos: {len(mejores_caracteres)} "
        f"(blockSize={mejor_block_size})"
    )

    return mejores_caracteres, crop_debug, mejor_binaria, mejor_block_size


# ==============================================================================
# 7. CONSTRUCCIÓN DE LAS FIGURAS PARA EL INFORME
# ==============================================================================

def generar_figuras(
    nombre_imagen,
    img_rgb,
    etapas_principales,
    etapas_alternativas,
    candidatos_principales,
    candidatos_alternativos,
    candidato_ganador,
    puntaje_maximo,
    mascara_azul,
    crop_rgb,
    crop_gray,
    binaria_caracteres,
    crop_debug,
    block_size,
    metodo_usado,
):
    """Crea y opcionalmente guarda tres láminas con las etapas del proceso."""

    # --------------------------------------------------------------------------
    # FIGURA 1: preprocesamiento y Canny
    # --------------------------------------------------------------------------
    fig1, axs1 = plt.subplots(2, 3, figsize=(15, 9))

    mostrar_en_eje(axs1[0, 0], img_rgb, "1. Imagen original")
    mostrar_en_eje(axs1[0, 1], etapas_principales["gris"], "2. Escala de grises", "gray")
    mostrar_en_eje(
        axs1[0, 2],
        etapas_principales["gaussiano_3x3"],
        "3. Filtro gaussiano 3×3",
        "gray",
    )
    mostrar_en_eje(axs1[1, 0], etapas_principales["sharpening"], "4. Sharpening", "gray")
    mostrar_en_eje(
        axs1[1, 1],
        etapas_principales["gaussiano_5x5"],
        "5. Filtro gaussiano 5×5",
        "gray",
    )
    mostrar_en_eje(axs1[1, 2], etapas_principales["canny"], "6. Bordes Canny", "gray")

    fig1.suptitle(f"Pipeline principal de detección — {nombre_imagen}", fontsize=16)
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    guardar_figura(fig1, f"{Path(nombre_imagen).stem}_01_preprocesamiento_canny.png")

    # --------------------------------------------------------------------------
    # FIGURA 2: detector alternativo y binarización de Otsu
    # --------------------------------------------------------------------------
    fig2, axs2 = plt.subplots(2, 3, figsize=(15, 9))

    mostrar_en_eje(axs2[0, 0], etapas_alternativas["blackhat"], "1. Transformación BlackHat", "gray")
    mostrar_en_eje(
        axs2[0, 1],
        etapas_alternativas["sobel_normalizado"],
        "2. Gradiente Sobel X",
        "gray",
    )
    mostrar_en_eje(
        axs2[0, 2],
        etapas_alternativas["sobel_suavizado"],
        "3. Sobel suavizado",
        "gray",
    )
    mostrar_en_eje(
        axs2[1, 0],
        etapas_alternativas["otsu"],
        f"4. Binarización Otsu (T={etapas_alternativas['umbral_otsu']:.0f})",
        "gray",
    )
    mostrar_en_eje(axs2[1, 1], etapas_alternativas["cierre"], "5. Clausura morfológica", "gray")
    mostrar_en_eje(axs2[1, 2], etapas_alternativas["dilatada"], "6. Dilatación", "gray")

    fig2.suptitle(f"Pipeline alternativo — {nombre_imagen}", fontsize=16)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    guardar_figura(fig2, f"{Path(nombre_imagen).stem}_02_otsu_morfologia.png")

    # --------------------------------------------------------------------------
    # FIGURA 3: candidatos, selección y segmentación de caracteres
    # --------------------------------------------------------------------------
    fig3, axs3 = plt.subplots(2, 3, figsize=(15, 9))

    img_candidatos_principales = dibujar_candidatos(img_rgb, candidatos_principales)
    img_candidatos_alternativos = dibujar_candidatos(
        img_rgb,
        candidatos_alternativos,
        color=(255, 0, 255),
    )
    img_ganador = dibujar_ganador(img_rgb, candidato_ganador, puntaje_maximo)

    mostrar_en_eje(
        axs3[0, 0],
        img_candidatos_principales,
        f"1. Candidatos Canny ({len(candidatos_principales)})",
    )
    mostrar_en_eje(
        axs3[0, 1],
        img_candidatos_alternativos,
        f"2. Candidatos alternativos ({len(candidatos_alternativos)})",
    )
    mostrar_en_eje(axs3[0, 2], mascara_azul, "3. Máscara HSV de color azul", "gray")
    mostrar_en_eje(
        axs3[1, 0],
        img_ganador,
        f"4. Candidato ganador ({metodo_usado})",
    )
    mostrar_en_eje(axs3[1, 1], binaria_caracteres, f"5. Umbral adaptativo (blockSize={block_size})", "gray")
    mostrar_en_eje(axs3[1, 2], crop_debug, "6. Segmentación de caracteres")

    fig3.suptitle(
        f"Selección y segmentación final — {nombre_imagen}",
        fontsize=16,
    )
    fig3.tight_layout(rect=[0, 0, 1, 0.96])
    guardar_figura(fig3, f"{Path(nombre_imagen).stem}_03_resultado_segmentacion.png")

    if not MOSTRAR_FIGURAS:
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)


# ==============================================================================
# 8. PROCESAMIENTO COMPLETO DE UNA IMAGEN
# ==============================================================================

def procesar_imagen(path):
    print("\n" + "=" * 70)
    print(f"Procesando: {path}")
    print("=" * 70)

    img_bgr, img_rgb, img_gray = cargar_imagen(path)
    img_h, img_w = img_gray.shape[:2]
    img_area = img_h * img_w

    # Pipeline principal
    edges, etapas_principales = preprocesar_principal(img_gray)
    candidatos_principales = extraer_candidatos(edges, img_area)

    # El pipeline alternativo se ejecuta siempre para poder mostrarlo en el informe.
    candidatos_alternativos, etapas_alternativas = extraer_candidatos_alternativo(
        img_gray,
        img_area,
    )

    candidatos_elegidos = candidatos_principales
    metodo_usado = "Canny"

    if candidatos_principales:
        puntajes, mascara_azul = puntuar_candidatos(
            candidatos_principales,
            img_gray,
            img_bgr,
        )
        idx_ganador = seleccionar_ganador(candidatos_principales, puntajes)
        puntaje_maximo = max(puntajes)
    else:
        puntajes = []
        mascara_azul = cv2.inRange(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV),
            AZUL_HSV_BAJO,
            AZUL_HSV_ALTO,
        )
        idx_ganador = None
        puntaje_maximo = -1

    # Se utiliza el alternativo si el principal no halló candidatos o si mejora
    # un resultado de puntaje bajo.
    evaluar_alternativo = not candidatos_principales or puntaje_maximo < 2

    if evaluar_alternativo and candidatos_alternativos:
        puntajes_alt, mascara_azul_alt = puntuar_candidatos(
            candidatos_alternativos,
            img_gray,
            img_bgr,
        )
        idx_alt = seleccionar_ganador(candidatos_alternativos, puntajes_alt)
        puntaje_alt = max(puntajes_alt)

        if not candidatos_principales or puntaje_alt > puntaje_maximo:
            candidatos_elegidos = candidatos_alternativos
            puntajes = puntajes_alt
            idx_ganador = idx_alt
            puntaje_maximo = puntaje_alt
            mascara_azul = mascara_azul_alt
            metodo_usado = "BlackHat + Sobel + Otsu"

    if not candidatos_elegidos or idx_ganador is None:
        print("  No se encontró ningún candidato válido.")
        return {
            "exito": False,
            "puntaje": 0,
            "caracteres": 0,
            "metodo": "Sin detección",
        }

    candidato_ganador = candidatos_elegidos[idx_ganador]
    x, y, w, h, ratio, area_rel, cnt = candidato_ganador

    crop_rgb = img_rgb[y:y + h, x:x + w]
    crop_gray = img_gray[y:y + h, x:x + w]

    caracteres, crop_debug, binaria_caracteres, block_size = segmentar_caracteres(
        crop_rgb,
        crop_gray,
    )

    print(f"  Método seleccionado: {metodo_usado}")
    print(f"  Puntaje máximo: {puntaje_maximo}/4")
    print(f"  Relación de aspecto del ganador: {ratio:.2f}")

    generar_figuras(
        nombre_imagen=Path(path).name,
        img_rgb=img_rgb,
        etapas_principales=etapas_principales,
        etapas_alternativas=etapas_alternativas,
        candidatos_principales=candidatos_principales,
        candidatos_alternativos=candidatos_alternativos,
        candidato_ganador=candidato_ganador,
        puntaje_maximo=puntaje_maximo,
        mascara_azul=mascara_azul,
        crop_rgb=crop_rgb,
        crop_gray=crop_gray,
        binaria_caracteres=binaria_caracteres,
        crop_debug=crop_debug,
        block_size=block_size,
        metodo_usado=metodo_usado,
    )

    return {
        "exito": True,
        "puntaje": puntaje_maximo,
        "caracteres": len(caracteres),
        "metodo": metodo_usado,
    }



procesar_imagen('tp2-pruebas/img_6.jpg')

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    resultados = {}

    print(f"Buscando imágenes en: {CARPETA_IMAGENES}")

    for numero in NUMEROS_IMAGENES:
        ruta_imagen = CARPETA_IMAGENES / f"img_{numero}.jpg"

        try:
            resultados[ruta_imagen.name] = procesar_imagen(ruta_imagen)
        except FileNotFoundError as error:
            print(f"  ERROR: {error}")
            resultados[ruta_imagen.name] = {
                "exito": False,
                "puntaje": 0,
                "caracteres": 0,
                "metodo": "Archivo inexistente",
            }

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"{'Imagen':<14} {'Éxito':>7} {'Puntaje':>9} {'Caracteres':>12}  Método")
    print("-" * 70)

    for nombre, resultado in resultados.items():
        exito = "SÍ" if resultado["exito"] else "NO"
        print(
            f"{nombre:<14} "
            f"{exito:>7} "
            f"{resultado['puntaje']:>9} "
            f"{resultado['caracteres']:>12}  "
            f"{resultado['metodo']}"
        )

    if GUARDAR_FIGURAS:
        print(f"\nLas figuras se guardaron en: {CARPETA_SALIDA}")

    if MOSTRAR_FIGURAS:
        plt.show()
