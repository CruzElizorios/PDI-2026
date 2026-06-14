import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot(img, title="Imagen", greyscale=False):

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cmap = 'gray' if greyscale else None
    plt.figure(figsize=(10, 6))
    plt.imshow(img_rgb, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

img_original = cv2.imread('tp2-pruebas/img_3.jpg')
print(f"Dimensiones de la imagen: {img_original.shape}")
plot(img_original)

# Convertir a escala de grises
img_gris = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
plot(img_gris, title="Imagen en Escala de Grises", greyscale=True)
print(f"Dimensiones de la imagen en gris: {img_gris.shape}")

thresh = cv2.threshold(img_gris, 120, 255, cv2.THRESH_BINARY_INV)[1]
plot(thresh, title="Imagen Binarizada", greyscale=True)

# Encontrar contornos (compatible con distintas versiones de OpenCV)
found = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contornos = found[0] if len(found) == 2 else found[1]

# Lienzo negro para dibujar sólo los contornos
canvas = np.zeros_like(img_original)
cv2.drawContours(canvas, contornos, -1, (0, 255, 0), 2)

# Superposición de contornos sobre la imagen original
overlay = img_original.copy()
cv2.drawContours(overlay, contornos, -1, (0, 255, 0), 2)

# Mostrar ambos: solo contornos y contornos sobre la imagen original
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[0].set_title('Contornos (solo)')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
axes[1].set_title('Contornos sobre original')
axes[1].axis('off')

plt.tight_layout()
plt.show()




#correr en todas las imagenes
for i in range(1, 13):
    img_path = f'tp2-pruebas/img_{i}.jpg'
    #print(f"\n📷 Procesando: {img_path}")
    img_original = cv2.imread(img_path)
    print(f"Dimensiones de la imagen: {img_original.shape}")
    #plot(img_original)

    # Convertir a escala de grises
    img_gris = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img_gris, 120, 255, cv2.THRESH_BINARY_INV)[1]
    plot(thresh, title=f"Imagen Binarizada {i}", greyscale=True)




import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carga y preprocesamiento básico (Unidad 1)
imgOriginal = cv2.imread('tp2-pruebas/img_7.jpg')
imgResultado = imgOriginal.copy()
imgGris = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2GRAY)

# Umbralado Adaptativo para aislar los bordes en autos claros (Unidad 3)
imgTh = cv2.adaptiveThreshold(imgGris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                             cv2.THRESH_BINARY_INV, 45, 10)

plot(imgTh, greyscale=True)
# Encontrar los contornos (Unidad 3 - Segmentación)
# Usamos RETR_EXTERNAL porque solo nos interesa la silueta exterior de la placa
contornos, _ = cv2.findContours(imgTh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Parámetros geométricos esperados (130mm / 400mm = 0.325)
# Damos tolerancia para absorber deformaciones por perspectiva tridimensional
relacionMinima = 0.22
relacionMaxima = 0.45
areaMinimaPatente = 100

for cnt in contornos:
    area = cv2.contourArea(cnt)
    
    if area > areaMinimaPatente:
        # Extraemos el rectángulo de área mínima orientado (con rotación)
        # Devuelve: ((centro_x, centro_y), (ancho, alto), angulo_rotacion)
        rectanguloOrientado = cv2.minAreaRect(cnt)
        (centro, dimensiones, angulo) = rectanguloOrientado
        anchoObjeto, altoObjeto = dimensiones
        
        # Evitamos divisiones por cero accidentales
        if anchoObjeto == 0 or altoObjeto == 0:
            continue
            
        # Como el rectángulo puede venir rotado, nos aseguramos de calcular 
        # la relación de aspecto siempre como (lado menor / lado mayor)
        ladoMenor = min(anchoObjeto, altoObjeto)
        ladoMayor = max(anchoObjeto, altoObjeto)
        relacionAspecto = ladoMenor / ladoMayor
        
        # Validamos si se encuentra en el rango esperado de la patente
        if relacionMinima <= relacionAspecto <= relacionMaxima:
            # Para poder dibujar un rectángulo rotado en OpenCV, 
            # calculamos las coordenadas de sus 4 esquinas exactas
            puntosEsquinas = cv2.boxPoints(rectanguloOrientado)
            puntosEsquinas = np.int0(puntosEsquinas) # Convertimos a enteros para pixelación
            
            # Dibujamos el contorno del recuadro sobre la copia a color
            # Usamos un color brillante (Verde: 0, 255, 0) y grosor 3
            cv2.drawContours(imgResultado, [puntosEsquinas], 0, (0, 255, 0), 3)

# Mostrar resultado final con colores RGB correctos (Unidad 1 y 5)
plt.figure(figsize=(10,6))
plt.imshow(cv2.cvtColor(imgResultado, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Patente Detectada mediante Contornos Orientados")
plt.show()