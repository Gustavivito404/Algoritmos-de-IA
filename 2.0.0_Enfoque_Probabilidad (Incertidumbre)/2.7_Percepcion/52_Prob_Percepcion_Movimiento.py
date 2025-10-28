# =========================================================
# 52 - MOVIMIENTO (Detección de cambio entre cuadros)
# ---------------------------------------------------------
# Descripción:
#   Detectamos movimiento comparando dos frames consecutivos.
#
#   Idea:
#     - Si algo se movió, los pixeles cambiaron.
#     - Podemos restar frame_t y frame_{t+1} para ver dónde hay diferencia.
#
#   Flujo clásico (visión estática con cámara fija):
#     1) Capturamos frame en t  (frame_1)
#     2) Capturamos frame en t+1 (frame_2)
#     3) diff = |frame_2 - frame_1|
#     4) Umbralizamos diff para quedarnos solo con los cambios fuertes
#        (zona que se movió).
#     5) Encontramos contornos y dibujamos bounding boxes donde hubo movimiento.
#
#   En este script:
#     - Generamos imágenes sintéticas (no video real)
#       con un rectángulo "que se desplaza".
#     - Aplicamos la detección de movimiento descrita.
#
#   Esto es la base de:
#     • detección de intrusos por cámara fija,
#     • conteo de objetos que pasan por una cinta,
#     • preprocesamiento para flujo óptico.
#
#   Librerías:
#     - OpenCV
#     - NumPy
#     - Matplotlib
# =========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Generar dos frames sintéticos
# ---------------------------------------------------------
alto, ancho = 200, 300

# frame_1: rectángulo en posición inicial
frame_1 = np.zeros((alto, ancho), dtype=np.uint8)
cv2.rectangle(frame_1, (60, 80), (130, 140), 255, thickness=-1)  # objeto blanco

# frame_2: el mismo rectángulo movido un poco a la derecha
frame_2 = np.zeros((alto, ancho), dtype=np.uint8)
cv2.rectangle(frame_2, (80, 80), (150, 140), 255, thickness=-1)  # desplazado +20 px en x

print("Frames sintéticos generados:")
print("  - frame_1: rectángulo blanco en (60,80)-(130,140)")
print("  - frame_2: mismo rectángulo desplazado a (80,80)-(150,140)")
print("  -> Simulamos un objeto que se mueve hacia la derecha\n")

# ---------------------------------------------------------
# 2. Diferencia absoluta entre frames
# ---------------------------------------------------------
# Donde frame_2 y frame_1 son distintos => posible movimiento.
diff_abs = cv2.absdiff(frame_2, frame_1)

print("Calculamos diferencia absoluta entre frames.")
print("  diff_abs = |frame_2 - frame_1|")
print(f"  Intensidad promedio del cambio: {np.mean(diff_abs):.2f}\n")

# ---------------------------------------------------------
# 3. Suavizado + Umbral
# ---------------------------------------------------------
# Suavizamos un poco para quitar ruido aislado
diff_blur = cv2.GaussianBlur(diff_abs, (5,5), sigmaX=0.8)

# Umbral binario: consideramos 'movimiento' los pixeles con cambio fuerte
_, movimiento_mask = cv2.threshold(diff_blur, 30, 255, cv2.THRESH_BINARY)

print("Aplicamos umbral para obtener máscara de movimiento.")
print("  Umbral = 30 (pixeles con cambio >30 se consideran movimiento)")
print(f"  Porcentaje de pixeles marcados como movimiento: "
      f"{100*np.mean(movimiento_mask>0):.2f}%\n")

# ---------------------------------------------------------
# 4. Encontrar contornos de las zonas en movimiento
# ---------------------------------------------------------
# Encontramos contornos sobre la máscara binaria
contornos, _ = cv2.findContours(movimiento_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("Contornos detectados en la máscara de movimiento:")
print(f"  Cantidad de regiones en movimiento: {len(contornos)}\n")

# Creamos una imagen en color para visualizar bounding boxes
frame_res = cv2.cvtColor(frame_2, cv2.COLOR_GRAY2BGR)

for c in contornos:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(frame_res, (x,y), (x+w,y+h), (0,255,255), 2)
    print(f"  Región en movimiento:")
    print(f"    Bounding box: x={x}, y={y}, w={w}, h={h}")
    print(f"    Área aprox: {w*h} pixeles\n")

# ---------------------------------------------------------
# 5. Visualización con matplotlib
# ---------------------------------------------------------
plt.figure(figsize=(12,6))

plt.subplot(2,3,1)
plt.imshow(frame_1, cmap="gray", vmin=0, vmax=255)
plt.title("Frame t")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(frame_2, cmap="gray", vmin=0, vmax=255)
plt.title("Frame t+1")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(diff_abs, cmap="gray", vmin=0, vmax=255)
plt.title("Diferencia Absoluta")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(movimiento_mask, cmap="gray", vmin=0, vmax=255)
plt.title("Máscara de Movimiento\n(umbralizada)")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(cv2.cvtColor(frame_res, cv2.COLOR_BGR2RGB))
plt.title("Detección de Movimiento\n(con bounding box)")
plt.axis("off")

plt.tight_layout()
plt.show()

# Comentario final:
#   • Donde el rectángulo cambió de lugar entre frames,
#     aparecen zonas blancas en diff_abs.
#
#   • La máscara binaria 'movimiento_mask' marca esas zonas.
#
#   • Con esas zonas podemos sacar contornos y dibujar
#     bounding boxes alrededor del objeto que se movió.
#
#   • Esto es exactamente la base de:
#       - detección de intrusos con cámara fija,
#       - segmentación de objetos en cinta transportadora,
#       - sistemas de conteo de personas/vehículos sencillos.
