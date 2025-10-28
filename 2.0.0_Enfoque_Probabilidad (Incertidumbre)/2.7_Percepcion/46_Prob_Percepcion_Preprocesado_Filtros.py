# =========================================================
# 46 - PREPROCESADO: FILTROS (Suavizado y Realce)
# ---------------------------------------------------------
# Descripción:
#   En percepción por computador, antes de detectar bordes,
#   segmentar o reconocer objetos, es común aplicar filtros
#   para reducir ruido o resaltar detalles importantes.
#
#   Este script muestra los filtros clásicos de preprocesado:
#
#   1) Filtro Promedio (Blur)
#   2) Filtro Gaussiano (Suavizado natural)
#   3) Filtro Mediana (Elimina ruido tipo "sal y pimienta")
#   4) Filtro de Realce (Sharpen)
#
#   Todo se aplica sobre una imagen sintética generada
#   por código: un gradiente + ruido aleatorio.
#
#   Librerías usadas:
#     - OpenCV (cv2) para filtros
#     - Matplotlib para visualización
#
#   Resultado:
#     Visualización lado a lado de cada filtro aplicado.
# =========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Generar imagen sintética (gradiente + ruido)
# ---------------------------------------------------------
alto, ancho = 300, 400
# Gradiente horizontal de 0 a 255
gradiente = np.tile(np.linspace(0, 255, ancho, dtype=np.uint8), (alto, 1))

# Agregar ruido aleatorio tipo “sal y pimienta”
ruido = np.random.randint(0, 50, (alto, ancho), dtype=np.uint8)
imagen_ruidosa = cv2.add(gradiente, ruido)

print("Imagen base generada:")
print(f"  Dimensiones: {imagen_ruidosa.shape}")
print(f"  Tipo de dato: {imagen_ruidosa.dtype}")
print(f"  Valor medio de brillo: {np.mean(imagen_ruidosa):.2f}\n")

# ---------------------------------------------------------
# 2. Aplicar diferentes filtros de preprocesado
# ---------------------------------------------------------

# (1) Filtro promedio
blur = cv2.blur(imagen_ruidosa, (5, 5))

# (2) Filtro gaussiano
gauss = cv2.GaussianBlur(imagen_ruidosa, (5, 5), sigmaX=1.0)

# (3) Filtro de mediana
mediana = cv2.medianBlur(imagen_ruidosa, 5)

# (4) Filtro de realce (sharpen)
# Usamos un kernel clásico que resalta bordes finos
kernel_sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)
sharpen = cv2.filter2D(imagen_ruidosa, -1, kernel_sharpen)

print("Filtros aplicados:")
print("  - Promedio (reduce ruido general)")
print("  - Gaussiano (suaviza sin perder tanto detalle)")
print("  - Mediana (elimina ruido impulsivo)")
print("  - Sharpen (realza bordes y contraste local)\n")

# ---------------------------------------------------------
# 3. Visualización con matplotlib
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))

plt.subplot(2, 3, 1)
plt.imshow(imagen_ruidosa, cmap="gray")
plt.title("Original + Ruido")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(blur, cmap="gray")
plt.title("Filtro Promedio (blur)")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(gauss, cmap="gray")
plt.title("Filtro Gaussiano")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(mediana, cmap="gray")
plt.title("Filtro Mediana")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(sharpen, cmap="gray")
plt.title("Filtro Sharpen")
plt.axis("off")

plt.tight_layout()
plt.show()

# Comentarios finales:
#   • El filtro promedio suaviza todo de forma uniforme.
#   • El gaussiano respeta mejor las transiciones suaves.
#   • El de mediana elimina puntos aislados de ruido.
#   • El de realce resalta los bordes y detalles finos.
#
#   En visión por computador, estos pasos se aplican
#   antes de la detección de bordes o segmentación.
