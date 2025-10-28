# =========================================================
# 47 - DETECCIÓN DE ARISTAS Y SEGMENTACIÓN
# ---------------------------------------------------------
# Descripción:
#   En visión computacional clásica, un paso crítico es
#   encontrar "bordes" o "aristas" en la imagen. Una arista
#   normalmente significa: hay un cambio brusco de intensidad.
#
#   Este script hace 3 cosas:
#
#   1) Genera una imagen sintética en escala de grises
#      con figuras geométricas (rectángulo, círculo, línea).
#
#   2) Calcula aristas usando:
#        - Gradiente Sobel en X
#        - Gradiente Sobel en Y
#        - Detector de Canny
#
#   3) Hace una segmentación MUY simple por umbral
#      (threshold binario), que separa "figura" vs "fondo".
#
#   Estas ideas son la base de:
#     • Detección de contornos
#     • Encontrar objetos en visión clásica
#     • Preprocesar para reconocimiento (OCR, piezas mecánicas, etc.)
#
#   Librerías:
#     - OpenCV (operadores de gradiente, Canny, threshold)
#     - Matplotlib (para visualizar resultados)
# =========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Generar imagen sintética con figuras bien definidas
# ---------------------------------------------------------
alto, ancho = 300, 400
img = np.zeros((alto, ancho), dtype=np.uint8)  # imagen en escala de grises (0-255)

# Dibujamos un rectángulo gris claro
# cv2.rectangle(img, esquina_sup_izq, esquina_inf_der, color, grosor)
cv2.rectangle(img, (50, 50), (180, 200), color=180, thickness=-1)

# Dibujamos un círculo gris medio
# cv2.circle(img, centro, radio, color, grosor)
cv2.circle(img, (280, 120), 50, color=130, thickness=-1)

# Dibujamos una línea diagonal blanca
# cv2.line(img, p1, p2, color, grosor)
cv2.line(img, (200, 220), (350, 280), color=255, thickness=4)

# Agregamos un poco de ruido leve para hacerlo más realista
ruido = np.random.randint(0, 25, (alto, ancho), dtype=np.uint8)
img_ruidosa = cv2.add(img, ruido)

print("Imagen sintética creada:")
print(f"  Dimensiones: {img.shape}")
print("  Incluye: rectángulo, círculo y línea diagonal.\n")

# ---------------------------------------------------------
# 2. Suavizado previo (preprocesado)
#    Antes de detectar bordes, se acostumbra a suavizar ruido.
# ---------------------------------------------------------
img_suavizada = cv2.GaussianBlur(img_ruidosa, (5, 5), sigmaX=1.0)

print("Aplicamos suavizado gaussiano para reducir ruido.\n")

# ---------------------------------------------------------
# 3. Gradientes de Sobel (detección de cambios de intensidad)
# ---------------------------------------------------------
# Sobel en X: detecta bordes verticales (cambios horizontales)
sobel_x = cv2.Sobel(img_suavizada, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3)
sobel_x_abs = cv2.convertScaleAbs(sobel_x)  # convertir a 0..255

# Sobel en Y: detecta bordes horizontales (cambios verticales)
sobel_y = cv2.Sobel(img_suavizada, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)
sobel_y_abs = cv2.convertScaleAbs(sobel_y)

# Magnitud combinada (opcional): sqrt(Sx^2 + Sy^2)
sobel_mag = cv2.magnitude(sobel_x, sobel_y)
sobel_mag_abs = cv2.convertScaleAbs(sobel_mag)

print("Calculamos gradiente Sobel en X, en Y y magnitud combinada.\n")

# ---------------------------------------------------------
# 4. Detector de Canny (bordes finos y binarios)
# ---------------------------------------------------------
# Canny hace:
#   - suavizado
#   - gradiente
#   - supresión no máxima
#   - histéresis de umbral
#
# Thresholds típicos: (low, high)
canny = cv2.Canny(img_suavizada, threshold1=50, threshold2=150)

print("Calculamos bordes con Canny.\n")

# ---------------------------------------------------------
# 5. Segmentación simple por umbral (thresholding)
# ---------------------------------------------------------
# La idea:
#   - Si el pixel es "brillante", lo marcamos como 255 (objeto)
#   - Si no, lo marcamos como 0 (fondo)
#
# cv2.threshold devuelve: valor_usado, imagen_binaria
_, segmentada_bin = cv2.threshold(img_suavizada, thresh=100, maxval=255, type=cv2.THRESH_BINARY)

print("Segmentación por umbral fijo aplicada (thresh=100).\n")

# ---------------------------------------------------------
# 6. Visualización con matplotlib
# ---------------------------------------------------------
plt.figure(figsize=(12, 8))

# Imagen original con ruido
plt.subplot(2, 3, 1)
plt.imshow(img_ruidosa, cmap="gray")
plt.title("Imagen sintética ruidosa")
plt.axis("off")

# Imagen suavizada
plt.subplot(2, 3, 2)
plt.imshow(img_suavizada, cmap="gray")
plt.title("Suavizada (Gaussiano)")
plt.axis("off")

# Sobel X
plt.subplot(2, 3, 3)
plt.imshow(sobel_x_abs, cmap="gray")
plt.title("Sobel X (bordes verticales)")
plt.axis("off")

# Sobel Y
plt.subplot(2, 3, 4)
plt.imshow(sobel_y_abs, cmap="gray")
plt.title("Sobel Y (bordes horizontales)")
plt.axis("off")

# Magnitud Sobel
plt.subplot(2, 3, 5)
plt.imshow(sobel_mag_abs, cmap="gray")
plt.title("Magnitud gradiente Sobel")
plt.axis("off")

# Canny
plt.subplot(2, 3, 6)
plt.imshow(canny, cmap="gray")
plt.title("Canny (bordes finales)")
plt.axis("off")

plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 7. Visualización de segmentación binaria
#    (separación figura / fondo)
# ---------------------------------------------------------
plt.figure(figsize=(5,4))
plt.imshow(segmentada_bin, cmap="gray")
plt.title("Segmentación binaria (umbral=100)")
plt.axis("off")
plt.show()

# Comentarios:
#   - Sobel X resalta bordes donde la intensidad cambia en X,
#     por ejemplo el borde izquierdo/derecho del rectángulo.
#
#   - Sobel Y resalta cambios verticales, como top/bottom.
#
#   - Canny entrega un mapa de bordes delgadito y binario,
#     ideal para detección de contornos.
#
#   - La 'segmentación' aquí es muy básica
#     (fondo oscuro vs objeto más brillante),
#     pero es la base de separar piezas en visión industrial.
#
#   Próximos pasos típicos:
#     - Encontrar contornos (cv2.findContours)
#     - Dibujar bounding boxes alrededor de cada objeto
#     - Medir área, radio, forma
