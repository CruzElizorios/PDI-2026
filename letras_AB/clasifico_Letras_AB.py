import cv2
from matplotlib import pyplot as plt
import numpy as np

# Defininimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)

# --- Cargo imagen --------------------------------------------------------------------------------
img = cv2.imread('letrasAB.png', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
imshow(img, title='Original')

# --- Filtro ruido impulsivo ----------------------------------------------------------------------
img_fil = cv2.medianBlur(img, 5)
imshow(img_fil, title='Filtro de Mediana')

# --- Paso a escala de grises ---------------------------------------------------------------------
img_fil_gray = cv2.cvtColor(img_fil, cv2.COLOR_RGB2GRAY)
imshow(img_fil_gray, title='Escala de Grises')

# --- Binarizo ------------------------------------------------------------------------------------
th, binary_img = cv2.threshold(img_fil_gray, 125, 1, cv2.THRESH_OTSU)
imshow(binary_img, title='Umbralado')

# --- Analisis ------------------------------------------------------------------------------------
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img) 
imshow(labels, title=f'Componentes conectadas ({num_labels})') # Por que son 22 en vez de 21?
stats

ii = 1
obj_area = stats[ii, cv2.CC_STAT_AREA]
obj_col_ini = stats[ii, cv2.CC_STAT_LEFT]
obj_col_end = stats[ii, cv2.CC_STAT_LEFT] + stats[ii, cv2.CC_STAT_WIDTH] - 1
obj_row_ini = stats[ii, cv2.CC_STAT_TOP]
obj_row_end = stats[ii, cv2.CC_STAT_TOP] + stats[ii, cv2.CC_STAT_HEIGHT] - 1

img_obj = (labels == ii).astype(np.uint8)
imshow(img_obj, title=f'label {ii} - area = {obj_area} - rows = [{obj_row_ini} {obj_row_end}] - cols = [{obj_col_ini} {obj_col_end}]', ticks=True)

# --- Opción 1: Elimino el objeto indeseado -------------------------------------------------------
binary_img_fil = binary_img.copy()
binary_img_fil[labels==1] = 0

# # --- Opción 2: Filtro por área -------------------------------------------------------------------
# binary_img_fil = binary_img.copy()
# AREA_TH = 10
# for i in range(num_labels):
#     if stats[i, cv2.CC_STAT_AREA] < AREA_TH:
#         binary_img_fil[labels==i] = 0

# # --- Opción 3: Operaciones morfológicas para mejorar la segmentación obtenida --------------------
# se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# binary_img_fil = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, se)   # Apertura para remover elementos pequeños
# binary_img_fil = cv2.morphologyEx(binary_img_fil, cv2.MORPH_CLOSE, se)  # Clausura para rellenar huecos.
# plt.figure(), plt.imshow(binary_img_fil, cmap='gray'), plt.title('Acondicionamiento Morfológico'), plt.show(block=False)

# --- Obtengo componentes conectados --------------------------------------------------------------
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img_fil)
imshow(labels, title=f'Componentes conectadas ({num_labels})') 

# *** OPCIONAL ****************************************************************
# Para visualizar mejor, observo los objetos en COLOR
# im_color = cv2.applyColorMap(labels.astype(np.uint8), cv2.COLORMAP_JET)     # El rango actual de valores es 0 a num_labels-1 (20).
#                                                                             # Como es tan chico el rango comparado al rango total [0 255], los colores son muy parecidos.
#                                                                             # Es conveniente llevar el rango de valores al rango [0 255] para generar colores mas diferentes entre si.
labels_color = np.uint8(255/(num_labels-1)*labels)                  # Llevo el rango de valores a [0 255] para diferenciar mejor los colores
# np.unique(labels_color)                                           # Por si quieren verificar los valores asignados....
im_color = cv2.applyColorMap(labels_color, cv2.COLORMAP_JET)
im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)                # El mapa de color que se aplica está en BGR --> convierto a RGB
imshow(im_color, title=f'Componentes conectadas ({num_labels})') # Por que son 22 en vez de 21?
# *****************************************************************************

# --- Clasificacion -------------------------------------------------------------------------------
labeled_shapes = np.zeros_like(img)
for i in range(1, num_labels):
    obj = (labels == i).astype(np.uint8)                                            # Genero una imagen que contenga solo el objeto "i"
    contours, _ = cv2.findContours(obj, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)     # Obtengo los contornos del objeto
    if len(contours) == 3:
        labeled_shapes[obj == 1, 2] = 255  # Si tiene 3 contornos --> B 
    else:
        labeled_shapes[obj == 1, 0] = 255  # Caso contrario --> A

imshow(labeled_shapes, title="Letras clasificadas")


