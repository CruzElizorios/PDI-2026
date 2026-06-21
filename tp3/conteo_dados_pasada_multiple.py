"""
TP3 - Cinco dados
Procesamiento de Imagenes I - TUIA / UNR

Versión con cambios mínimos:
  PASADA 1 - Detecta intervalo de reposo.
  PASADA 2 - Cuenta valores durante todo el reposo, pero NO dibuja.
  PASADA 3 - Dibuja solamente los valores finales consolidados por moda.

Objetivo:
  Evitar que aparezcan etiquetas incorrectas en frames intermedios.
"""

import cv2
import numpy as np
from collections import Counter


# ---------------------------------------------------------------------------
# CONFIGURACIÓN  velocidades
#  ---------------------------------------------------------------------------

FACTOR_LENTO = 2.5
# 1.0 = velocidad original
# 2.0 = video dura el doble
# 2.5 = más lento
# 3.0 = todavía más lento


# ---------------------------------------------------------------------------
# SEGMENTACION
# ---------------------------------------------------------------------------
def segmentar_cartulina(frame):
    """Otsu sobre el canal 'a' de LAB. Verde (a bajo) -> blanco. Devuelve mascara."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    a = cv2.split(lab)[1]
    _, mask_verde = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return mask_verde


def obtener_roi(mask_verde):
    """ROI = componente blanca mas grande (la cartulina), excluyendo el fondo."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask_verde)
    if num <= 1:
        return None
    label_cartulina = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    roi = np.uint8(labels == label_cartulina) * 255
    return roi


def detectar_dados(frame, roi):
    """
    Dados = huecos internos del ROI (contornos hijos).
    Se filtran por tamaño relativo a la mediana y por color (rojo en canal 'a').
    Devuelve la lista de contornos de dados.
    """
    contornos, jerarquia = cv2.findContours(roi, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if jerarquia is None:
        return []

    huecos = [
        c for c, h in zip(contornos, jerarquia[0])
        if h[3] != -1 and cv2.contourArea(c) > 500
    ]

    if not huecos:
        return []

    lados = np.array([
        (cv2.boundingRect(c)[2] + cv2.boundingRect(c)[3]) / 2
        for c in huecos
    ])

    mediana = np.median(lados)

    huecos = [
        c for c, l in zip(huecos, lados)
        if 0.6 * mediana <= l <= 1.4 * mediana
    ]

    if not huecos:
        return []

    a = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))[1]

    NEUTRO, MARGEN = 128, 10
    umbral_rojo = NEUTRO + MARGEN

    dados = []

    for c in huecos:
        mask = np.zeros(a.shape, np.uint8)
        cv2.drawContours(mask, [c], -1, 255, cv2.FILLED)

        if cv2.mean(a, mask=mask)[0] > umbral_rojo:
            dados.append(c)

    return dados


# ---------------------------------------------------------------------------
# CONTEO DE PUNTOS
# ---------------------------------------------------------------------------
def contar_puntos(frame, contorno_dado):
    """
    Cuenta los puntos blancos de un dado.
    Criterios: S bajo (desaturado) AND V alto (brillante) + area relativa + circularidad.
    """
    x, y, w, h = cv2.boundingRect(contorno_dado)
    crop = frame[y:y + h, x:x + w]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    umbral_s, _ = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    umbral_v, _ = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    puntos_mask = np.zeros(S.shape, np.uint8)
    puntos_mask[(S < umbral_s) & (V > umbral_v)] = 255

    contornos, _ = cv2.findContours(puntos_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    area_dado = w * h
    area_min, area_max = 0.010 * area_dado, 0.025 * area_dado

    valor = 0

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)

        if perimetro == 0:
            continue

        circularidad = 4 * np.pi * area / (perimetro ** 2)

        if area_min <= area <= area_max and circularidad > 0.7:
            valor += 1

    return valor


def moda_valores(lista):
    """
    Devuelve el valor más repetido ignorando ceros.
    Esto evita etiquetas transitorias incorrectas.
    """
    validos = [v for v in lista if 1 <= v <= 6]

    if not validos:
        return 0

    return Counter(validos).most_common(1)[0][0]


# ---------------------------------------------------------------------------
# SEÑAL DE MOVIMIENTO
# ---------------------------------------------------------------------------
def mascara_dados(frame):
    """Mascara binaria con los dados rellenos + cuantos hay. Para la señal de movimiento."""
    mask_verde = segmentar_cartulina(frame)
    roi = obtener_roi(mask_verde)

    salida = np.zeros(mask_verde.shape, np.uint8)

    if roi is None:
        return salida, 0

    dados = detectar_dados(frame, roi)

    for c in dados:
        cv2.drawContours(salida, [c], -1, 255, cv2.FILLED)

    return salida, len(dados)


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------
def procesar_video(ruta_entrada, ruta_salida, n_dados=5, n_frames_reposo=2,
                   frac_umbral_mov=0.09):

    cap = cv2.VideoCapture(ruta_entrada)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir {ruta_entrada}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"\n=== Procesando {ruta_entrada} ===")
    print(f"    {width}x{height} @ {fps}fps")

    # -----------------------------------------------------------------------
    # PASADA 1: detectar reposo
    # -----------------------------------------------------------------------
    movimiento = []
    cantidad = []
    mask_prev = None

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        mask, n = mascara_dados(frame)
        cantidad.append(n)

        if mask_prev is not None:
            movimiento.append(int(np.count_nonzero(cv2.absdiff(mask, mask_prev))))
        else:
            movimiento.append(0)

        mask_prev = mask

    cap.release()

    pico = max(movimiento) if movimiento else 1
    umbral_mov = pico * frac_umbral_mov

    print(f"    Pico de movimiento: {pico} | umbral quieto: {umbral_mov:.0f}")

    en_reposo = []
    quietos_seguidos = 0

    for i in range(len(movimiento)):
        quieto = (cantidad[i] == n_dados) and (movimiento[i] < umbral_mov)

        if quieto:
            quietos_seguidos += 1
        else:
            quietos_seguidos = 0

        en_reposo.append(quietos_seguidos >= n_frames_reposo)

    frames_reposo = [i for i, r in enumerate(en_reposo) if r]

    if frames_reposo:
        print(
            f"    Reposo: frames {frames_reposo[0]}-{frames_reposo[-1]} "
            f"({len(frames_reposo)} frames)"
        )
    else:
        print("    No se detecto reposo.")

    # -----------------------------------------------------------------------
    # PASADA 2: contar valores finales, sin dibujar
    # -----------------------------------------------------------------------
    valores_por_dado = [[] for _ in range(n_dados)]

    cap = cv2.VideoCapture(ruta_entrada)
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        if idx < len(en_reposo) and en_reposo[idx]:

            mask_verde = segmentar_cartulina(frame)
            roi = obtener_roi(mask_verde)
            dados = detectar_dados(frame, roi) if roi is not None else []

            if len(dados) == n_dados:

                dados = sorted(
                    dados,
                    key=lambda c: cv2.boundingRect(c)[0]
                )

                for i, c in enumerate(dados):
                    valor = contar_puntos(frame, c)
                    valores_por_dado[i].append(valor)

        idx += 1

    cap.release()

    valores_finales = [
        moda_valores(vs)
        for vs in valores_por_dado
    ]

    resultado = ", ".join(
        f"D{i + 1}-{v}"
        for i, v in enumerate(valores_finales)
    )

    print(f"    [VALORES FINALES] {resultado}")

    # -----------------------------------------------------------------------
    # PASADA 3: dibujar solamente valores finales
    # -----------------------------------------------------------------------
    fps_salida = max(1, int(fps / FACTOR_LENTO))

    out = cv2.VideoWriter(
        ruta_salida,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps_salida,
        (width, height)
    )

    cap = cv2.VideoCapture(ruta_entrada)
    idx = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        vis = frame.copy()

        if idx < len(en_reposo) and en_reposo[idx]:

            mask_verde = segmentar_cartulina(frame)
            roi = obtener_roi(mask_verde)
            dados = detectar_dados(frame, roi) if roi is not None else []

            if len(dados) == n_dados:

                dados = sorted(
                    dados,
                    key=lambda c: cv2.boundingRect(c)[0]
                )

                for i, c in enumerate(dados, start=1):
                    x, y, w, h = cv2.boundingRect(c)

                    valor = valores_finales[i - 1]

                    cv2.rectangle(
                        vis,
                        (x, y),
                        (x + w, y + h),
                        (0, 0, 255),
                        3
                    )

                    cv2.putText(
                        vis,
                        f"D{i}-{valor}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        4
                    )

        out.write(vis)
        idx += 1

    cap.release()
    out.release()

    print(f"    Guardado: {ruta_salida}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    videos = [
        "tp3/videos/tirada_1.mp4",
        "tp3/videos/tirada_2.mp4",
        "tp3/videos/tirada_3.mp4",
        "tp3/videos/tirada_4.mp4"
    ]

    for v in videos:
        salida = v.replace(".mp4", "_out.mp4")
        procesar_video(v, salida)