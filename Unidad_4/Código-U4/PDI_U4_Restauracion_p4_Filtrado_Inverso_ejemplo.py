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

# Definimos una función para generar el kernel que modela el efecto de borrosidad lineal
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

# Definimos una función que implementa el Filtrado Inverso de Wiener
def weiner_deconvolution(img, psf, snr):
    """
    Aplica una deconvolución Wiener dada la imagen degradada,
    el kernel o PSF que modela la distorción (Point Spread Function) y  un valor de SNR.

    Entradas
    ----------
        img : numpy.ndarray
            La imagen degradada
        psf : numpy.ndarray
            Kernel o PSF (Point Spread Function) que modela la degradación convolucional.
        snr : float
            Relación señal a ruido, en db.
    Salida
    -------
            La imagen restaurada : numpy.ndarray
    """
    # NSR: relación noise/signal (el inverso del SNR), en escala lineal.
    NSR = 10.0 ** (-0.1 * snr)

    # Expandir el kernel con ceros para igualar el tamaño de la imagen.
    psf_pad = np.zeros_like(img)
    kh, kw = psf.shape
    psf_pad[:kh, :kw] = psf

    # Deconvolución Wiener en el dominio frecuencial.
    IMG = np.fft.fft2(img)
    PSF = np.fft.fft2(psf_pad)
    RES = IMG * (PSF / (np.abs(PSF)**2 + NSR))

    # Transformada inversa.
    deconvolved = np.fft.ifft2(RES)

    # Dado que el elemento central del kernel está ubicado en (kw/2,kh/2) y no(0,0),
    # la imagen resultante sufre una traslación de (kw/2,kh/2).
    # Esto se resuelve aplicando un roll (desplazamiento estilo buffer circular) en (-kw/2,-kh/2).
    deconvolved = np.roll(deconvolved, -kh//2, 0)
    deconvolved = np.roll(deconvolved, -kw//2, 1)
    return deconvolved

# -----------------------------------------------------------------------------------------
# --- Ejemplo 1: Restauración de imagen degradada con Modelo Lineal de Borrosidad ---------
# -----------------------------------------------------------------------------------------

# Cargamos la imagen y visualizamos
f = cv2.imread("patente_2_blur.jpg")
f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)

print(f.shape)
print(np.unique(f))
imshow(f, title="Imagen con borrosidad", colorbar=False)

"""
Al observar la imagen, notamos que está degradada por un Modelo Lineal de Borrosidad.
Por lo tanto, para poder restaurar esta imagen utilizando el Filtrado Inverso de Wiener, 
debemos encontrar los parámetros que definen la PSF.
Haciendo zoom en la imagen original, encontramos correspondencia entre dos puntso y en 
base a ellos, calculamos el ángulo y el desplazamiento de la PSF.
"""
P1= (247,203)
P2= (277,187)

f2 = f.copy()
cv2.circle(f2,P1,2,(255,0,0),-1)
cv2.circle(f2,P2,2,(255,0,0),-1)
imshow(f2, title="Puntos seleccionados", colorbar=False)

dx = P2[0] - P1[0]
dy = np.abs(P2[1] - P1[1])
angle = np.arctan(dy/dx)
angle_grad = 180*angle/np.pi
d = (dx**2 + dy**2)**0.5
print(f"Ángulo: {angle_grad:5.2f}  -  Desplazamiento: {d:5.2f}")

# ---  Filtrado Inverso de Wiener ------------------------------------------
# Primero, acondicionamos la imagen para evitar problemas en los cálculos matemáticos
fs = np.float32(f)/255.0   # Paso a float y normalizo el máximo rango a [0.0  1.0].
print(np.unique(fs))

# Luego definimos la PSF en base al análisis realizado sobre la imagen deteriorada

# largo = 34
# angulo = 28
largo = 30
angulo = 30
psf = motion_blur_kernel(length=largo, rotation=angulo)
imshow(psf, title="Kernel utilizado para el Filtrado Inverso")

# Finalmente, utilizamos el Filtrado Inverso de Wiener (paramétrico) para restaurar la imagen degradada.
snr = 33                            # Definimos manualmente el ruido (utilizar rangos entre 0 y 50)
f_restored = np.zeros_like(fs)      # Inicializo la imagen de salida
print(f_restored.shape)
print(f_restored.dtype)

for n in range(3):
    result = weiner_deconvolution(fs[:, :, n], psf, snr)    # Aplico Wiener
    f_restored[:,:,n] = np.real(result)                     # Transformo a nros. reales

print(np.unique(f_restored))
imshow(f_restored, colorbar=False, title="Imagen Restaurada mediante el Filtrado Inverso de Wiener")

# Acondiciono la imagen resultante
f_restored_sc = cv2.convertScaleAbs(f_restored,alpha=255)
np.unique(f_restored_sc)
imshow(f_restored_sc, colorbar=False, title="Imagen Restaurada mediante el Filtrado Inverso de Wiener - Scaled")