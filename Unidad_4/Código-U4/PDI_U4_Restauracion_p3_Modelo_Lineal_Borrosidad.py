import cv2
import numpy as np
import matplotlib.pyplot as plt

# Definimos función para mostrar imágenes
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

# --- Cargo Imagen ------------------------------------------------------------
img = cv2.imread('cameraman.tif', cv2.IMREAD_GRAYSCALE)
# img = cv2.imread('checkboard.tif', cv2.IMREAD_GRAYSCALE)
img.shape
imshow(img, title="Imagen Original", ticks=True)
np.unique(img)

# --- Modelo Lineal de Borrosidad ---------------------------------------------
# Genero kernel 
k = 35
w = np.zeros((k,k))
# w[k//2,:] = 1/k                 # Horizontal
# w[:,k//2] = 1/k                 # Vertical
# w = np.diag(np.ones((k)))/k     # Diagonal
w.sum()

# Filtro
img_b = cv2.filter2D(img, cv2.CV_64F, w)
img_b.dtype
np.unique(img_b)
img_bs = cv2.convertScaleAbs(img_b)

# Muestro
plt.figure()
plt.subplot(121), imshow(w, title='Kernel', new_fig=False, ticks=True)
plt.subplot(122), imshow(img_bs, title='Imagen con borrosidad lineal + escalado', new_fig=False)
plt.show(block=False)

plt.figure()
plt.subplot(121), imshow(img_b, title=f'Borrosidad lineal (min={img_b.min():5.2f} | max={img_b.max():5.2f})', new_fig=False)
plt.subplot(122), imshow(img_bs, title=f'Borrosidad lineal + escalado (min={img_bs.min()} | max={img_bs.max()})', new_fig=False)
plt.show(block=False)

# --- Modelo Lineal de borrosidad - Rotacion ---------------------------------------------
# Genero kernel 
k = 31
w = np.zeros((k,k))
w[k//2,:] = 1                 # Horizontal
rot = 45
M = cv2.getRotationMatrix2D((k//2, k//2), rot, 1)
w_rot = cv2.warpAffine(w, M, w.shape)
w_rot /= w_rot.sum()
w_rot.sum()
np.unique(w_rot)

# Filtro
img_b = cv2.filter2D(img, cv2.CV_64F, w_rot)
img_b.dtype
np.unique(img_b)
img_bs = cv2.convertScaleAbs(img_b)

# Muestro
plt.figure()
plt.subplot(121), imshow(w_rot, title='Kernel', new_fig=False, ticks=True)
plt.subplot(122), imshow(img_bs, title='Imagen con borrosidad lineal + escalado', new_fig=False)
plt.show(block=False)

plt.figure()
plt.subplot(121), imshow(img_b, title=f'Borrosidad lineal (min={img_b.min():5.2f} | max={img_b.max():5.2f})', new_fig=False)
plt.subplot(122), imshow(img_bs, title=f'Borrosidad lineal + escalado (min={img_bs.min()} | max={img_bs.max()})', new_fig=False)
plt.show(block=False)

# Defino una función para crear los kernels de manera genérica
def motion_blur_kernel(length, rotation):
    """
    Genera un kernel de borrosidad lineal.

    Parametros
    ----------
        length : int      --> La longitud del filtro
        rotation : float  --> El ángulo del desplazamiento, en grados.
    """
    # --- Genero kernel horizontal -----------------------------------
    w = np.zeros((length, length))
    w[length//2,:] = 1          

    # --- Roto el kernel de acuerdo al angulo ------------------------
    M = cv2.getRotationMatrix2D((length//2, length//2), rotation, 1)
    M_rot = cv2.warpAffine(w, M, w.shape)
    M_rot /= M_rot.sum()

    return M_rot

L = 27
ANG = 15
w = motion_blur_kernel(L, ANG)
img_b = cv2.filter2D(img, cv2.CV_64F, w)

plt.figure()
plt.subplot(121), imshow(w, title=f'Kernel - length = {L} | rotation={ANG}', new_fig=False, ticks=True)
plt.subplot(122), imshow(img_b, title='Imagen con borrosidad lineal', new_fig=False)
plt.show(block=False)
