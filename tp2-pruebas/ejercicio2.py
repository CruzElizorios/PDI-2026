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


# 1. Cargar imagen
img_original = cv2.imread('tp2-pruebas/img_7.jpg')
print(f"Dimensiones de la imagen: {img_original.shape}")
plot(img_original)

# Crear una copia para dibujar el recuadro de color (y no alterar la original)
img_resultado = img_original.copy()

# Convertir a escala de grises
img_gris = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
plot(img_gris, title="Imagen en Escala de Grises", greyscale=True)
print(f"Dimensiones de la imagen en gris: {img_gris.shape}")

# 2. Umbralización (usamos Otsu para un cálculo automático del umbral óptimo)
# Se asume que la patente es clara (blanca) y resaltará. 
_, img_th = cv2.threshold(img_gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
plot(img_th, title="Imagen Binarizada", greyscale=True)
# Nota: Si la placa se binariza al revés (negro sobre blanco), 
# podrías usar cv2.THRESH_BINARY_INV.


# En lugar de cv2.threshold global, usamos el enfoque regional adaptativo:
# blockSize=45 (debe ser un número impar grande que cubra parte de la patente)
# C=10 (constante que se resta a la media para limpiar ruido)
img_th = cv2.adaptiveThreshold(img_gris, 255, 
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 25, 15)
plot(img_th, title="Imagen Binarizada", greyscale=True)



# 3. Extraer componentes conectadas
# connectivity=8 evalúa píxeles adyacentes en diagonales también
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_th, connectivity=8, ltype=cv2.CV_32S)

# 4. Filtrar y dibujar
# Definimos la tolerancia para el cociente (alto/ancho = 130/400 = 0.325)
# Rango sugerido considerando perspectiva: [0.20, 0.45]
relacion_min = 0.15
relacion_max = 0.30

# funciona para autos oscuros y patente blanca sin invertir (THRESH_BINARY_INV)
relacion_min = 0.20
relacion_max = 0.45


# Opcional pero recomendado: un área mínima para evitar procesar ruido diminuto
area_minima = 500 
area_maxima = 20000

for i in range(1, num_labels): # Iniciamos en 1 para ignorar la etiqueta 0 (fondo)
    
    # Extraer estadísticas de la componente actual
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    ancho = stats[i, cv2.CC_STAT_WIDTH]
    alto = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    
    # Filtrado básico por área para descartar basura
    if area > area_minima and area < area_maxima:
        
        # Calcular el cociente (Relación de Aspecto = alto / ancho)
        cociente = alto / float(ancho)
        
        # Verificar si cumple con el rango geométrico de la patente
        if relacion_min <= cociente <= relacion_max:
            
            # Dibujar un recuadro de color verde (BGR: 0, 255, 0) de grosor 3
            cv2.rectangle(img_resultado, (x, y), (x + ancho, y + alto), (0, 255, 0), 3)

# Mostrar la imagen final con el recuadro detectado
plt.figure(figsize=(10,6))
plt.imshow(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
plt.title("Detección de Patente")
plt.axis('off')
plt.show()



plt.figure(figsize=(10,6))
#plt.imshow(img_original)
#plt.imshow(img_gris,cmap='gray')
plt.imshow(img_th)
#plt.imshow(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
plt.title("Detección de Patente")
plt.axis('off')
plt.show()

def lectorPatentes(img_path, relacion_min=0.20, relacion_max=0.45, area_minima=500, area_maxima=20000, mostrar=True):
    """
    Detecta y marca patentes en una imagen.
    
    Args:
        img_path: Ruta de la imagen
        relacion_min: Relación altura/ancho mínima esperada para la patente
        relacion_max: Relación altura/ancho máxima esperada para la patente
        area_minima: Área mínima del componente conectado
        area_maxima: Área máxima del componente conectado
        mostrar: Si True, muestra la imagen con detecciones
    
    Returns:
        dict con: 
            - 'img_original': imagen original
            - 'img_resultado': imagen con detecciones marcadas
            - 'detecciones': lista de bounding boxes detectadas [(x, y, ancho, alto), ...]
            - 'exito': True si se cargó la imagen correctamente
    """
    resultado = {
        'exito': False,
        'img_original': None,
        'img_resultado': None,
        'detecciones': [],
        'archivo': img_path
    }
    
    # Intentar cargar la imagen
    try:
        img_original = cv2.imread(img_path)
        if img_original is None:
            print(f"❌ Error: No se pudo cargar la imagen: {img_path}")
            return resultado
    except Exception as e:
        print(f"❌ Error al cargar {img_path}: {str(e)}")
        return resultado
    
    resultado['img_original'] = img_original
    img_resultado = img_original.copy()
    
    # Convertir a escala de grises
    img_gris = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    
    # Umbralización con Otsu
    _, img_th = cv2.threshold(img_gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Extraer componentes conectadas
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_th, connectivity=8, ltype=cv2.CV_32S)
    
    # Filtrar y dibujar detecciones
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        ancho = stats[i, cv2.CC_STAT_WIDTH]
        alto = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filtrado básico por área
        if area > area_minima and area < area_maxima:
            # Calcular relación de aspecto
            cociente = alto / float(ancho)
            
            # Verificar si cumple con el rango geométrico
            if relacion_min <= cociente <= relacion_max:
                # Dibujar recuadro verde
                cv2.rectangle(img_resultado, (x, y), (x + ancho, y + alto), (0, 255, 0), 3)
                resultado['detecciones'].append((x, y, ancho, alto))
    
    resultado['img_resultado'] = img_resultado
    resultado['exito'] = True
    
    # Mostrar resultado si se solicita
    if mostrar:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Original: {img_path}")
        axes[0].axis('off')
        
        axes[1].imshow(cv2.cvtColor(img_resultado, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"Detecciones encontradas: {len(resultado['detecciones'])}")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return resultado


# Ejemplo de uso de la función
print("=" * 60)
print("PROCESANDO IMÁGENES CON DETECTOR DE PATENTES")
print("=" * 60)

# Definir rango de imágenes a procesar
imagenes_a_procesar = [1,2,3, 4, 5, 6, 7, 8, 9, 10]  # Puedes ajustar este rango

#4, 5, 6, 7, 8, 9, 10
resultados = []
for i in imagenes_a_procesar:
    img_path = f'tp2-pruebas/img_{i}.jpg'
    print(f"\n📷 Procesando: {img_path}")
    resultado = lectorPatentes(img_path, relacion_min=0.15, relacion_max=0.30, area_minima=500, area_maxima=20000, mostrar=True)
    
    if resultado['exito']:
        print(f"✅ Patentes detectadas: {len(resultado['detecciones'])}")
        for idx, (x, y, w, h) in enumerate(resultado['detecciones'], 1):
            print(f"   Detección {idx}: posición ({x}, {y}), tamaño {w}x{h}")
        resultados.append(resultado)
    else:
        print(f"⚠️  No se procesó correctamente")

print(f"\n{'=' * 60}")
print(f"Procesamiento completado: {len(resultados)}/{len(imagenes_a_procesar)} imágenes")
print(f"{'=' * 60}")