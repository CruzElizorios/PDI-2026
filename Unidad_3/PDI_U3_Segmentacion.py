import cv2
import numpy as np
import matplotlib.pyplot as plt

# Defininimos función para mostrar imágenes
def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=True, ticks=False):
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

# -------------------------------------------------------------------------------
# --- Detección de puntos -------------------------------------------------------
# -------------------------------------------------------------------------------
f = cv2.imread('punto.tif', cv2.IMREAD_GRAYSCALE)   # Leemos imagen
imshow(f, title="Imagen con un punto oculto")

w = -1*np.ones((3,3))                   # Definimos el kernel para...
w[1,1] = 8                              # ... detectar puntos
fp = cv2.filter2D(f, cv2.CV_64F, w)     # Filtramos
np.unique(fp)
imshow(fp, title="Filtrado con kernel para detectar Puntos")

fpn = abs(fp)                           # Acondicionamiento: abs()
# fpn = cv2.convertScaleAbs(fp)             # En este caso, no generaría problemas. Pero ver el ejemplo siguiente.
np.unique(fpn)                          
imshow(fpn, title="Filtrado + abs()")

f_point = fpn > fpn.max()*0.9           # Detección mediante umbralado
imshow(f_point, title="Puntos Segmentados")
np.sum(f_point)                         # Mostramos cuantos puntos detectamos

plt.figure()
ax = plt.subplot(221)
imshow(f,new_fig=False, title="Imagen Original")
plt.subplot(222, sharex=ax, sharey=ax), imshow(fp, new_fig=False, title="Filtrado")
plt.subplot(223, sharex=ax, sharey=ax), imshow(fpn, new_fig=False, title="Filtrado + abs()")
plt.subplot(224, sharex=ax, sharey=ax), imshow(f_point, new_fig=False, title="Filtrado + abs() + Umbralado")
plt.suptitle("Detección de Punto")
plt.show(block=False)

# -------------------------------------------------------------------------------
# --- Detección de lineas orientadas --------------------------------------------
# -------------------------------------------------------------------------------
f = cv2.imread('lineas_orientadas.tif', cv2.IMREAD_GRAYSCALE)   # Leemos imagen
f.shape
np.unique(f)
imshow(f, ticks=True, title="Imagen con lineas")

# Filtramos
w = np.array([[-1,-1,-1],[2,2,2],[-1,-1,-1]])     # Horizontal
# w = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]])       # Vertical
# w = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]])       # +45º
# w = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]])       # -45º
f_fil = cv2.filter2D(f, cv2.CV_64F, w)      
np.unique(f_fil)
imshow(f_fil, title="Filtrado con kernel para detectar lineas orientadas")

f_fil_abs = abs(f_fil)                          # Acondicionamos
# f_fil_abs = cv2.convertScaleAbs(f_fil)        # Así generaría problemas
np.unique(f_fil_abs)
imshow(f_fil_abs, title="Filtrado + abs()")

f_det = f_fil_abs >= f_fil_abs.max()*1.0    # Detección mediante umbralado
imshow(f_det, title="Lineas segmentadas")

plt.figure()
ax = plt.subplot(221)
imshow(f, new_fig=False, title="Imagen Original")
plt.subplot(222, sharex=ax, sharey=ax), imshow(f_fil, new_fig=False, title="Filtrado")
plt.subplot(223, sharex=ax, sharey=ax), imshow(f_fil_abs, new_fig=False, title="Filtrado + abs()")
plt.subplot(224, sharex=ax, sharey=ax), imshow(f_det, new_fig=False, title="Filtrado + abs() + Umbralado")
plt.suptitle("Detección de Lineas Orientadas")
plt.show(block=False)

# -------------------------------------------------------------------------------
# --- Detección de lineas orientadas - Otro Ejemplo -----------------------------
# -------------------------------------------------------------------------------
f = cv2.imread('circuito.tif', cv2.IMREAD_GRAYSCALE)   # Leemos imagen
f.shape
np.unique(f)
imshow(f, ticks=True)

# --- Version 1 ------------------------------------------
w = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]])   # +45º
f_fil = cv2.filter2D(f, cv2.CV_64F, w)          # Filtramos
f_fil_acond = abs(f_fil)                        # Acondicionamos: abs()           
f_det = f_fil_acond >= f_fil_acond.max()*1.0    # Detección mediante umbralado

np.unique(f_fil)
np.unique(f_fil_acond)

plt.figure()
ax = plt.subplot(221)
imshow(f, new_fig=False, title="Imagen Original", ticks=True)
plt.subplot(222, sharex=ax, sharey=ax), imshow(f_fil, new_fig=False, title="Filtrado", ticks=True)
plt.subplot(223, sharex=ax, sharey=ax), imshow(f_fil_acond, new_fig=False, title="Filtrado + Acondicionamiento", ticks=True)
plt.subplot(224, sharex=ax, sharey=ax), imshow(f_det, new_fig=False, title="Filtrado + abs() + Umbralado", ticks=True)
plt.suptitle("Detección de Lineas Orientadas")
plt.show(block=False)

# --- Version 2 (un poco mejor para este caso) --------------------------------
w = np.array([[-1,-1,2],[-1,2,-1],[2,-1,-1]])   # +45º
f_fil = cv2.filter2D(f, cv2.CV_64F, w)          # Filtramos
f_fil_acond = f_fil.copy()                      # Acondicionamos: A los valores < 0...
f_fil_acond[f_fil_acond<0] = 0                  # ... les asigno 0
f_det2 = f_fil_acond >= f_fil_acond.max()*0.9    # Detección mediante umbralado

np.unique(f_fil)
np.unique(f_fil_acond)

plt.figure()
ax = plt.subplot(221)
imshow(f, new_fig=False, title="Imagen Original", ticks=True)
plt.subplot(222, sharex=ax, sharey=ax), imshow(f_fil, new_fig=False, title="Filtrado", ticks=True)
plt.subplot(223, sharex=ax, sharey=ax), imshow(f_fil_acond, new_fig=False, title="Filtrado + Acondicionamiento", ticks=True)
plt.subplot(224, sharex=ax, sharey=ax), imshow(f_det2, new_fig=False, title="Filtrado + abs() + Umbralado", ticks=True)
plt.suptitle("Detección de Lineas Orientadas")
plt.show(block=False)

# --- Comparo ambas versiones ------------------------------------------------
plt.figure()
ax = plt.subplot(121); imshow(f_det, new_fig=False, title="Version 1", colorbar=False)
plt.subplot(122, sharex=ax, sharey=ax), imshow(f_det2, new_fig=False, title="Version 2", colorbar=False)
plt.show(block=False)

# -------------------------------------------------------------------------------
# --- Detección de Bordes - Gradiente -------------------------------------------
# -------------------------------------------------------------------------------
# Cargo Imagen 
f = cv2.imread('building.tif', cv2.IMREAD_GRAYSCALE)
f.dtype
f.shape
imshow(f)

# Filtrado
ddepth = cv2.CV_16S  # Formato salida
grad_x = cv2.Sobel(f, ddepth, 1, 0, ksize=3) # https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d
grad_y = cv2.Sobel(f, ddepth, 0, 1, ksize=3) # Tutorial: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html
np.unique(grad_x)
np.unique(grad_y)

# Acondicionamiento
abs_grad_x = cv2.convertScaleAbs(grad_x)                        # abs() + casting uint8
abs_grad_y = cv2.convertScaleAbs(grad_y)                        #
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)     # Combinamos.  https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html

plt.figure()
ax = plt.subplot(221)
imshow(f, new_fig=False, title="Imagen Original", ticks=True)
plt.subplot(222, sharex=ax, sharey=ax), imshow(abs_grad_x, new_fig=False, title="Sobel x", ticks=True)
plt.subplot(223, sharex=ax, sharey=ax), imshow(abs_grad_y, new_fig=False, title="Sobel y", ticks=True)
plt.subplot(224, sharex=ax, sharey=ax), imshow(grad, new_fig=False, title="Gradiente", ticks=True)
plt.show(block=False)

# Umbralizamos los gradientes 
abs_grad_x_th = np.zeros_like(abs_grad_x) 
abs_grad_x_th[abs_grad_x == abs_grad_x.max()] = 255

abs_grad_y_th = np.zeros_like(abs_grad_y) 
abs_grad_y_th[abs_grad_y == abs_grad_y.max()] = 255

grad_th = np.zeros_like(grad) 
grad_th[grad >= 0.5*grad.max()] = 255

plt.figure()
ax = plt.subplot(221)
imshow(abs_grad_x, new_fig=False, title="Sobel x")
plt.subplot(222, sharex=ax, sharey=ax), imshow(abs_grad_x_th, new_fig=False, title="Sobel x + Umbralado")
plt.subplot(223, sharex=ax, sharey=ax), imshow(abs_grad_y, new_fig=False, title="Sobel y")
plt.subplot(224, sharex=ax, sharey=ax), imshow(abs_grad_y_th, new_fig=False, title="Sobel y + Umbralado")
plt.show(block=False)

plt.figure()
ax = plt.subplot(121)
imshow(grad, new_fig=False, title="Gradiente")
plt.subplot(122, sharex=ax, sharey=ax), imshow(grad_th, new_fig=False, title="Gradiente + Umbralado")
plt.show(block=False)

plt.figure()
ax = plt.subplot(121)
imshow(f, new_fig=False, title="Imagen Original", ticks=True)
plt.subplot(122, sharex=ax, sharey=ax), imshow(grad_th, new_fig=False, title="Gradiente + Umbralado")
plt.show(block=False)


# -------------------------------------------------------------------------------
# --- Detección de Bordes - Gradiente - Sobel - Versión estricta y aproximada----
# -------------------------------------------------------------------------------
# Cargo Imagen 
f = cv2.imread('building.tif', cv2.IMREAD_GRAYSCALE)
imshow(f)

# Filtrado
ddepth = cv2.CV_64F  # Formato salida
grad_x = cv2.Sobel(f, ddepth, 1, 0, ksize=3) # https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758cb6f375ce782c8d
grad_y = cv2.Sobel(f, ddepth, 0, 1, ksize=3) # Tutorial: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_gradients/py_gradients.html

grad = np.sqrt(grad_x**2 + grad_y**2)       # Exacta
grad_aprox = grad_x**2 + grad_y**2          # Aproximada
np.unique(grad)
np.unique(grad_aprox)

grad_n = cv2.convertScaleAbs(grad)
grad_aprox_n = cv2.convertScaleAbs(grad_aprox)
np.unique(grad_n)
np.unique(grad_aprox_n)

plt.figure()
ax = plt.subplot(221)
imshow(grad, new_fig=False, title="Gradiente - Magnitud")
plt.subplot(222, sharex=ax, sharey=ax), imshow(grad_n, new_fig=False, title="Gradiente - Magnitud + Casting")
plt.subplot(223, sharex=ax, sharey=ax), imshow(grad_aprox, new_fig=False, title="Gradiente - Magnitud aprox.")
plt.subplot(224, sharex=ax, sharey=ax), imshow(grad_aprox_n, new_fig=False, title="Gradiente - Magnitud aprox. + Casting")
plt.show(block=False)

grad_th = grad >= grad.max()*0.25
# grad_aprox_th = grad_aprox >= grad_aprox.max()*0.25 # Analizar...
# grad_aprox_th = grad_aprox >= grad_aprox.max()*0.05 # Analizar...
grad_aprox_th = grad_aprox >= grad_aprox.max()*0.0625 # Así se obtiene casi el mismo resultado que con grad y th=0.25
plt.figure()
ax = plt.subplot(221)
imshow(grad, new_fig=False, title="Gradiente - Magnitud")
plt.subplot(222, sharex=ax, sharey=ax), imshow(grad_th, new_fig=False, title="Gradiente - Magnitud + Umbralado")
plt.subplot(223, sharex=ax, sharey=ax), imshow(grad_aprox, new_fig=False, title="Gradiente - Magnitud aprox.")
plt.subplot(224, sharex=ax, sharey=ax), imshow(grad_aprox_th, new_fig=False, title="Gradiente - Magnitud aprox. + Umbralado")
plt.show(block=False)


# --- Comparo con la versión anterior ------------------------------------
grad2_x = cv2.Sobel(f, ddepth, 1, 0, ksize=3)
grad2_y = cv2.Sobel(f, ddepth, 0, 1, ksize=3)
abs_grad2_x = cv2.convertScaleAbs(grad2_x)                        
abs_grad2_y = cv2.convertScaleAbs(grad2_y)                        
grad2 = cv2.addWeighted(abs_grad2_x, 0.5, abs_grad2_y, 0.5, 0)

plt.figure()
ax = plt.subplot(221)
imshow(f, new_fig=False, title="Imagen Original")
plt.subplot(222, sharex=ax, sharey=ax), imshow(grad_n, new_fig=False, title="Gradiente - Magnitud + Casting")
plt.subplot(223, sharex=ax, sharey=ax), imshow(grad_aprox_n, new_fig=False, title="Gradiente - Magnitud aprox. + Casting")
plt.subplot(224, sharex=ax, sharey=ax), imshow(grad2, new_fig=False, title="Gradiente version 2")
plt.show(block=False)

grad2_th = grad2 >= grad2.max()*0.51
plt.figure()
ax = plt.subplot(321)
imshow(grad, new_fig=False, title="Gradiente - Magnitud")
plt.subplot(322, sharex=ax, sharey=ax), imshow(grad_th, new_fig=False, title="Gradiente - Magnitud + Umbralado")
plt.subplot(323, sharex=ax, sharey=ax), imshow(grad_aprox, new_fig=False, title="Gradiente - Magnitud aprox.")
plt.subplot(324, sharex=ax, sharey=ax), imshow(grad_aprox_th, new_fig=False, title="Gradiente - Magnitud aprox. + Umbralado")
plt.subplot(325, sharex=ax, sharey=ax), imshow(grad2, new_fig=False, title="Gradiente version 2")
plt.subplot(326, sharex=ax, sharey=ax), imshow(grad2_th, new_fig=False, title="Gradiente version 2 + Umbralado")
plt.show(block=False)


# -------------------------------------------------------------------------------
# --- Detección de Bordes - Laplacian of Gaussian (LoG) -------------------------
# -------------------------------------------------------------------------------
f = cv2.imread('building.tif', cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(f, (3,3), 0)                # Aplicamos un filtro pasa-bajos (eliminamos ruido...)
LoG = cv2.Laplacian(blur, cv2.CV_64F, ksize=3)      # Obtenemos el Laplaciano
LoG_abs = cv2.convertScaleAbs(LoG)                  # Acondicionamiento (abs + cast)
np.unique(LoG)
np.unique(LoG_abs)
imshow(LoG_abs, title="LoG + convertScaleAbs()")
# imshow(np.abs(LoG), title="LoG + abs()")          # Para analizar...

LoG_abs_th = LoG_abs > LoG_abs.max()*0.3
# LoG_abs_th = LoG_abs > LoG_abs.max()*0.45
imshow(LoG_abs_th, title="LoG + convertScaleAbs() + umbral")

# Definimos función para encontrar los cruces por cero
def Zero_crossing(image):
    z_c_image = np.zeros(image.shape)
    # For each pixel, count the number of positive and negative pixels in the neighborhood
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [image[i+1, j-1],image[i+1, j],image[i+1, j+1],image[i, j-1],image[i, j+1],image[i-1, j-1],image[i-1, j],image[i-1, j+1]]
            d = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h>0:
                    positive_count += 1
                elif h<0:
                    negative_count += 1
            # If both negative and positive values exist in the pixel neighborhood, then that pixel is a potential zero crossing
            
            z_c = ((negative_count > 0) and (positive_count > 0))
            
            # Change the pixel value with the maximum neighborhood difference with the pixel
            if z_c:
                if image[i,j]>0:
                    z_c_image[i, j] = image[i,j] + np.abs(e)
                elif image[i,j]<0:
                    z_c_image[i, j] = np.abs(image[i,j]) + d
                
    # Normalize and change datatype to 'uint8' (optional)
    z_c_norm = z_c_image/z_c_image.max()*255
    z_c_image = np.uint8(z_c_norm)

    return z_c_image

# Calculamos los cruces por cero del LoG, umbralamos y visualizamos
LoG_z = Zero_crossing(LoG)
th = LoG_z.max()*0.3
LoG_zth = np.uint8(LoG_z > th)*255

plt.figure()
ax = plt.subplot(121); imshow(LoG_z, new_fig=False, title="LoG + Zero Corssing", colorbar=False)
plt.subplot(122, sharex=ax, sharey=ax), imshow(LoG_zth, new_fig=False, title="LoG + Zero Corssing + Umbral", colorbar=False)
plt.show(block=False)


# --- Componentes conectadas -----------------------------------------------------------------------
img = cv2.imread('objects.tif', cv2.IMREAD_GRAYSCALE)
img.shape
img.dtype
np.unique(img)
imshow(img, ticks=True)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8, ltype=cv2.CV_32S)  # https://docs.opencv.org/4.5.3/d3/dc0/group__imgproc__shape.html#ga107a78bf7cd25dec05fb4dfc5c9e765f
# num_labels: Cantidad de elementos
# labels: Matriz con etiquetas
# stats: Matriz de estadisticas de los elementos (bounding box + area)
# centroids: Centroides de elementos
num_labels
stats
centroids
labels
np.unique(labels)
imshow(labels)

# --- Otra opcion ----------------
# output = cv2.connectedComponentsWithStats(img, connectivity=8, ltype=cv2.CV_32S)
# num_labels = output[0]  # Cantidad de elementos
# labels = output[1]      # Matriz con etiquetas
# stats = output[2]       # Matriz de stats
# centroids = output[3]   # Centroides de elementos
# --------------------------------

# Coloreamos los elementos
labels_norm = np.uint8( (255/(num_labels-1)) * labels)    # Cambiamos el rango:
print(np.unique(labels_norm))                             # 0 a num_labels -->  0 a 255 (con paso 255/num_labels)
imshow(labels_norm)

# im_color = cv2.applyColorMap(labels_norm, cv2.COLORMAP_JET)   # Aplico un mapa de color
im_color = cv2.applyColorMap(labels_norm, cv2.COLORMAP_HOT)     # Aplico un mapa de color
imshow(im_color, color_img=True, colorbar=False)                    # Observar los colores ---> Están en BGR...

im_color = cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB)    # ... Paso a RGB
imshow(im_color, color_img=True, colorbar=False)

for centroid in centroids:  # Dibujo los centroides
    cv2.circle(im_color, tuple(np.int32(centroid)), 9, color=(255,255,255), thickness=-1)
for st in stats:            # Dibujo los bounding-boxes
    cv2.rectangle(im_color, (st[0], st[1]), (st[0]+st[2], st[1]+st[3]), color=(0,255,0), thickness=2)
imshow(img=im_color, color_img=True, colorbar=False)

# --- CANNY ---------------------------------------------------------------------------------------
f = cv2.imread('building.tif', cv2.IMREAD_GRAYSCALE)
imshow(f)		 

f_blur = cv2.GaussianBlur(f, ksize=(3, 3), sigmaX=1.5)
plt.figure()
ax = plt.subplot(121)
imshow(f, new_fig=False, title="Imagen Original", ticks=True)
plt.subplot(122, sharex=ax, sharey=ax), imshow(f_blur, new_fig=False, title="Gaussian Blur")
plt.show(block=False)

gcan = cv2.Canny(f_blur, threshold1=80, threshold2=120)
gcan.dtype
np.unique(gcan)
imshow(gcan, title="Canny")					 

gcan1 = cv2.Canny(f_blur, threshold1=0.04*255, threshold2=0.1*255)
gcan2 = cv2.Canny(f_blur, threshold1=0.4*255, threshold2=0.5*255)
gcan3 = cv2.Canny(f_blur, threshold1=0.4*255, threshold2=0.75*255)
imshow(gcan1)

plt.figure()
ax = plt.subplot(221)
imshow(f, new_fig=False, title="Imagen Original")
plt.subplot(222, sharex=ax, sharey=ax), imshow(gcan1, new_fig=False, title="Canny - U1=4% | U2=10%")
plt.subplot(223, sharex=ax, sharey=ax), imshow(gcan2, new_fig=False, title="Canny - U1=40% | U2=50%")
plt.subplot(224, sharex=ax, sharey=ax), imshow(gcan3, new_fig=False, title="Canny - U1=40% | U2=75%")
plt.show(block=False)

# --- Contornos -----------------------------------------------------------------------------------
f = cv2.imread('contornos.png', cv2.IMREAD_GRAYSCALE)             # Leemos imagen
f.dtype
np.unique(f)
imshow(f, colorbar=False)

# Tutorial: https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
contours, hierarchy = cv2.findContours(f, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga95f5b48d01abc7c2e0732db24689837b
type(contours)
len(contours)
type(contours[0])
contours[0].shape
contours[0][0]
contours[0][0].shape
contours[0][0][0]
contours[0][0][0].shape

type(hierarchy)
hierarchy.shape
hierarchy
hierarchy[0][0,:]

# Dibujamos todos los contornos
fc = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
cv2.drawContours(fc, contours, contourIdx=-1, color=(0, 0, 255), thickness=2)  # https://docs.opencv.org/3.4/d6/d6e/group__imgproc__draw.html#ga746c0625f1781f1ffc9056259103edbc
imshow(fc, colorbar=False, title="Contornos")

# Contornos externos
contours, hierarchy = cv2.findContours(f, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
fc = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
cv2.drawContours(fc, contours, contourIdx=-1, color=(0, 255, 0), thickness=2)
imshow(fc, colorbar=False, title="Contornos Externos")

# Contornos por jerarquía
contours, hierarchy = cv2.findContours(f, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print(hierarchy)    # hierarchy: [Next, Previous, First_Child, Parent]

# --- Contornos que no tienen madres/padres ----------------------------------------------
fc = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
for ii in range(len(contours)):
    if hierarchy[0][ii][3]==-1:
        cv2.drawContours(fc, contours, contourIdx=ii, color=(0, 255, 0), thickness=2)
imshow(fc, colorbar=False, title="Contornos sin madre/padre")

# --- Contornos que no tienen hijas/os ----------------------------------------------
fc = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
for ii in range(len(contours)):
    if hierarchy[0][ii][2]==-1:
        cv2.drawContours(fc, contours, contourIdx=ii, color=(0, 255, 0), thickness=2)
imshow(fc, colorbar=False, title="Contornos sin hijas/os")

# --- Ejemplo particular ---------------------------------------------------------
# Analizamos un ejemplo particular: seleccionamos un contorno y dibujamos su madre/padre, hijas/os, etc.
k = 4
fc = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
cv2.drawContours(fc, contours, contourIdx=k, color=(0, 255, 0), thickness=2)
hierarchy[0][k]
imshow(fc, colorbar=False, title="Contorno particular")

# Dibujo al madre/padre en rojo
if hierarchy[0][k][3] != -1:
    cv2.drawContours(fc, contours, contourIdx=hierarchy[0][k][3], color=(255, 0, 0), thickness=2)
imshow(fc, colorbar=False, title="Contorno particular")

# Dibujo los hijas/os en azul
for ii in range(len(contours)):
    if hierarchy[0][ii][3]==k:
        cv2.drawContours(fc, contours, contourIdx=ii, color=(0, 0, 255), thickness=2)
imshow(fc, colorbar=False, title="Contorno particular")

# Dibujo todos los que están en su mismo nivel (hermanas/os)
for ii in range(len(contours)):
    if hierarchy[0][ii][3]==hierarchy[0][k][3]:
        cv2.drawContours(fc, contours, contourIdx=ii, color=(0, 255, 255), thickness=2)
imshow(fc, colorbar=False, title="Contorno particular")


# --- Ordeno según los contornos mas grandes -------------------------------------
contours_area = sorted(contours, key=cv2.contourArea, reverse=True)
fc = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
cv2.drawContours(fc, contours_area, contourIdx=0, color=(255, 0, 0), thickness=2)
cv2.drawContours(fc, contours_area, contourIdx=1, color=(0, 255, 0), thickness=2)
cv2.drawContours(fc, contours_area, contourIdx=2, color=(0, 0, 255), thickness=2)
imshow(fc, colorbar=False, title="Contornos ordenados por area")

# --- Aproximación de contornos desde cv2.findContours() -----------------------
contours, hierarchy = cv2.findContours(f, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours_aprox, hierarchy_aprox = cv2.findContours(f, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
ii = 2

cont = contours[ii]
cont.shape
N_cont = cont.shape[0]

cont_aprox = contours_aprox[ii]
cont_aprox.shape
N_cont_aprox = cont_aprox.shape[0]

fc = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
cv2.drawContours(fc, contours, contourIdx=ii, color=(0, 255, 0), thickness=1)

fc_aprox = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
cv2.drawContours(fc_aprox, contours_aprox, contourIdx=ii, color=(0, 255, 0), thickness=1)
for p in contours_aprox[ii]:
    cv2.circle(fc_aprox, tuple(np.int32(p[0])), radius=7, color=(255,0,0), thickness=-1)

plt.figure()
ax = plt.subplot(121); imshow(fc, new_fig=False, color_img=True, colorbar=False, title=f"Contorno sin aproximar - {N_cont} puntos")
plt.subplot(122, sharex=ax, sharey=ax), imshow(fc_aprox, new_fig=False, colorbar=False, color_img=True, title=f"Contorno aproximado - {N_cont_aprox} puntos")
plt.suptitle("Aproximación de contornos desde cv2.findContours()")
plt.show(block=False)

# -- Aproximación de contornos con polinomios ----------------------------------
# Los contornos también se pueden aproximar, en vez de obtener todos los pixels 
# que lo componenen. Primero seleccionemos un contorno cualquiera...

# cnt = contours[2] # Rectángulo
cnt = contours[12]  # Círculo
fc = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
cv2.drawContours(fc, cnt, contourIdx=-1, color=(255, 0, 0), thickness=2)
imshow(fc, colorbar=False, title="Contorno particular")

approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)   # https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga0012a5fdaea70b8a9970165d98722b4c
len(cnt)
len(approx)

fc = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
cv2.drawContours(fc, approx, contourIdx=-1, color=(0, 0, 255), thickness=2)
imshow(fc, colorbar=False, title="Contorno particular APROXIMADO")

# Bounding Box
x,y,w,h = cv2.boundingRect(cnt)
fc = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
cv2.drawContours(fc, cnt, contourIdx=-1, color=(255, 0, 0), thickness=2)
cv2.rectangle(fc, (x,y), (x+w,y+h), color=(255, 0, 255), thickness=2)
imshow(fc, colorbar=False, title="Contorno particular y su bounding-box")

# Momentos del contorno
M=cv2.moments(cnt)
huMoments = cv2.HuMoments(M)  # https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gab001db45c1f1af6cbdbe64df04c4e944

# --- Hough Lineas --------------------------------------------------------------------------------
# Tutorial:
#   https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
#   https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
f = cv2.imread('contornos.png', cv2.IMREAD_GRAYSCALE)
imshow(f, colorbar=False, title="Imagen Original")
f.dtype
np.unique(f)

# Primero, detectamos los bordes de la imagen
edges = cv2.Canny(f, 100, 170, apertureSize=3)
imshow(edges, colorbar=False, title="Detección de bordes")

# Detecto líneas utilizando el método de Hough cv2.HoughLines()
# https://docs.opencv.org/3.4/dd/d1a/group__imgproc__feature.html#ga46b4e588934f6c8dfd509cc6e0e4545a
lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=250)
# lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=170)
# lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=115)
print(type(lines))
print(lines.shape)
print(lines[0,:,:])
print(lines[0][0])
print(lines[0][0].shape)

# --- Dibujo las lineas detectadas ----------------
# VERSION 1
f_lines = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
for i in range(0, len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    a=np.cos(theta)
    b=np.sin(theta)
    x0=a*rho
    y0=b*rho
    x1=int(x0+1000*(-b))
    y1=int(y0+1000*(a))
    x2=int(x0-1000*(-b))
    y2=int(y0-1000*(a))
    cv2.line(f_lines, (x1,y1), (x2,y2), (0,255,0), 2)
imshow(f_lines, colorbar=False, title=f"Lineas detectadas ({len(lines)})")  # Visualizo

# VERSION 2
x = np.arange(0,f.shape[1])
imshow(f, colorbar=False, title=f"Lineas detectadas ({len(lines)})", new_fig=True)    
for i in range(0, len(lines)):
    rho = lines[i][0][0]
    theta = lines[i][0][1]
    y = (rho - x*np.cos(theta)) / (np.sin(theta) + np.finfo(np.float64).eps) # Importante sumar eps!
    plt.plot(x, y, '-r')
plt.ylim([f.shape[0]-1,0])    
plt.show(block=False)

# --- Hough Circulos --------------------------------------------------------------------------------
# Tutorial: https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html
# https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
f = cv2.imread('logo_opencv.png', cv2.IMREAD_GRAYSCALE)
imshow(f, colorbar=False, title="Imagen Original")

# Aplicamos borrosidad
fb = cv2.medianBlur(f, 5)
plt.figure()
ax = plt.subplot(121)
imshow(f, new_fig=False, title="Imagen Original", ticks=True)
plt.subplot(122, sharex=ax, sharey=ax), imshow(fb, new_fig=False, title="Imagen con borrosidad")
plt.show(block=False)

# Detectamos círculos utilizando el método de Hough cv2.HoughCircles()
#
circles = cv2.HoughCircles(fb, method=cv2.HOUGH_GRADIENT, dp=1, minDist=10)                            # Valores iniciales: No detecta nada. Posible solución: disminuir umbral de votos
circles = cv2.HoughCircles(fb, method=cv2.HOUGH_GRADIENT, dp=1, minDist=10, param2=20)                 # Si bajamos mucho el umbral de votos --> Muchisimos círculos!
circles = cv2.HoughCircles(fb, method=cv2.HOUGH_GRADIENT, dp=1, minDist=10, param2=40)                   # Ajustamos umbral de votos --> Detecto solo círculos chicos
circles = cv2.HoughCircles(fb, method=cv2.HOUGH_GRADIENT, dp=1, minDist=10, param2=40, minRadius=30)   # Ajustamos radio mínimo --> Detecto solo círculos grandes

print(type(circles))
print(circles.shape)
print(circles[0,:,:])
Nc = circles.shape[1]
print(circles[0,0,:])

# Dibujo los círculos detectados
circles = np.uint16(np.around(circles))
fc = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
for i in circles[0,:]:
    cv2.circle(fc, (i[0],i[1]), i[2], (0,255,0), 2)   # Dibujo el círculo
    cv2.circle(fc, (i[0],i[1]), 2, (0,0,255), 2)      # Dibujo el centro del círculo

# Visualizo
imshow(fc, colorbar=False, title=f"Círculos detectados ({Nc})")


# -------------------------------------------------------------------------------
# --- Umbralado -----------------------------------------------------------------
# -------------------------------------------------------------------------------
img = cv2.imread("text.png",cv2.IMREAD_GRAYSCALE)
print(img.shape)
print(img.dtype)
print(np.unique(img))
imshow(img, title="Imagen Original", colorbar=False)

# Intentamos umbralar definiendo manualmente el umbral
th = 100
img_th = img>th                 # De esta manera, obtenemos una matriz booleana: True/False
print(img_th.dtype)
print(np.unique(img_th))

img_th = np.uint8(img>th)*255   # Generalmente, queremos uint8: 0/255
print(img_th.dtype)
print(np.unique(img_th))

imshow(img_th, title="Umbralado manual", colorbar=False)

# Otra forma es utilizar la función  cv2.threshold()
# https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gae8a4a146d1ca78c626a53577199e9c57
th = 100
th_out, img_th = cv2.threshold(img, thresh=th, maxval=255, type=cv2.THRESH_BINARY)
print(th_out)
print(img_th.dtype)
print(np.unique(img_th))

imshow(img_th, title="Umbralado manual", colorbar=False)

# --- Selección automática del umbral -------------------------------------------
T = (img.min() + img.max())/2                           # Inicializamos el valor de T
flag = False                                            # Inicializamos el flag para iterar
while not flag:
   g = img >= T                                         # Umbralamos
   Tnext = 0.5*(np.mean(img[g]) + np.mean(img[~g]))     # Definimos nuevo umbral
   flag = np.abs(T-Tnext) < 0.5                         # Verificamos si hay cambios apreciables en el nuevo umbral
   T = Tnext
print(T)
_, img_th_aut = cv2.threshold(img, thresh=T, maxval=255, type=cv2.THRESH_BINARY)
imshow(img_th_aut, title="Umbralado automático", colorbar=False)

# --- Otsu --------------------------------------------
T_otsu, img_th_otsu = cv2.threshold(img, thresh=127, maxval=255, type=cv2.THRESH_OTSU)
print(T_otsu)

# --- Graficas -----------------------------------------
plt.figure()
ax1 = plt.subplot(221); plt.xticks([]), plt.yticks([]), plt.imshow(img, cmap="gray"), plt.title('Imagen Original')
plt.subplot(222,sharex=ax1,sharey=ax1), plt.imshow(~img_th_aut, cmap="gray"), plt.title(f'Umbral Automático: {T:5.2f}')
plt.subplot(223,sharex=ax1,sharey=ax1), plt.imshow(~img_th_otsu, cmap="gray"), plt.title(f'Umbral Otsu: {T_otsu:5.2f}')
plt.show(block=False)

# --- Ejercicios ----------------------------------------
f = cv2.imread("01-coins1.pgm")
imshow(f, color_img=True, colorbar=False, title="Monedas")

f = cv2.imread("ojos.png")
f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
imshow(f, color_img=True, colorbar=False, title="Ojos")     

