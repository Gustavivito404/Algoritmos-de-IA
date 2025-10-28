# =========================================================
# 51 - ETIQUETADO DE LÍNEAS / COMPONENTES CONECTADAS
# ---------------------------------------------------------
# Descripción:
#   El etiquetado de componentes conectadas (CCL) busca
#   identificar y numerar todas las "regiones blancas"
#   en una imagen binaria.
#
#   Ejemplo:
#       Imagen binaria -> varios parches blancos separados.
#       Salida -> cada parche recibe una etiqueta distinta:
#                 objeto 1, objeto 2, objeto 3, ...
#
#   Este proceso es la base de:
#     • Contar piezas en visión industrial.
#     • Encontrar blobs en segmentación.
#     • Localizar objetos para medirlos.
#
#   En este script:
#     1) Creamos una imagen binaria sintética con varias figuras:
#        - rectángulos
#        - círculos
#        - puntos gruesos
#     2) Aplicamos cv2.connectedComponents() para etiquetar.
#     3) Coloreamos cada etiqueta con un color aleatorio
#        (para visualizar mejor).
#     4) Calculamos información de cada objeto:
#        - número de pixeles (área)
#        - bounding box (min/max x,y)
#
#   Nota:
#     Este tipo de recorrido históricamente se llama
#     "etiquetado de líneas" porque el algoritmo clásico
#     recorre la imagen fila por fila (scanline), asignando
#     y unificando etiquetas.
#
#   Requisitos:
#     - OpenCV
#     - NumPy
#     - Matplotlib
# =========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Generar imagen binaria sintética
# ---------------------------------------------------------
alto, ancho = 300, 400
img_bin = np.zeros((alto, ancho), dtype=np.uint8)  # fondo negro (0)

# Dibujar blobs blancos (valor 255)
# Rectángulo relleno
cv2.rectangle(img_bin, (30, 30), (120, 120), 255, thickness=-1)

# Círculo relleno
cv2.circle(img_bin, (250, 80), 40, 255, thickness=-1)

# Otro rectángulo más delgado
cv2.rectangle(img_bin, (200, 180), (360, 210), 255, thickness=-1)

# Pequeña mancha aislada
cv2.circle(img_bin, (70, 220), 15, 255, thickness=-1)

# Líneas gruesas cruzadas
cv2.line(img_bin, (280, 200), (320, 260), 255, thickness=10)
cv2.line(img_bin, (320, 200), (280, 260), 255, thickness=10)

print("Imagen binaria sintética creada.")
print("  - Varias regiones blancas separadas sobre fondo negro.")
print(f"  Dimensiones: {img_bin.shape}\n")

# ---------------------------------------------------------
# 2. Etiquetado de componentes conectadas
# ---------------------------------------------------------
# connectedComponents:
#   Entrada:
#     - imagen binaria (0=fondo, >0=objeto)
#   Salida:
#     - num_labels: cuántas etiquetas (incluye el fondo como 0)
#     - labels: imagen donde cada píxel tiene la etiqueta de su región
#
# Nota:
#   connectivity=8 usa vecindarios de 8 pixeles (diagonal también cuenta)
num_labels, labels = cv2.connectedComponents(img_bin, connectivity=8)

print("Etiquetado completado.")
print(f"  Número total de etiquetas (incluye fondo): {num_labels}")
print("  Etiqueta 0 = fondo negro\n")

# ---------------------------------------------------------
# 3. Asignar color a cada etiqueta para visualizar
# ---------------------------------------------------------
# Creamos una imagen RGB para mostrar cada componente
img_color = np.zeros((alto, ancho, 3), dtype=np.uint8)

# Colores aleatorios para cada etiqueta (excepto el fondo)
rng = np.random.default_rng(seed=1)
etiqueta_a_color = {
    0: (0, 0, 0)  # fondo negro
}
for lab in range(1, num_labels):
    # Color aleatorio en BGR
    color_bgr = rng.integers(low=50, high=255, size=3, dtype=np.uint8)
    etiqueta_a_color[lab] = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))

# Pintar cada píxel según su etiqueta
for y in range(alto):
    for x in range(ancho):
        lab = labels[y, x]
        img_color[y, x] = etiqueta_a_color[lab]

# Convertimos BGR->RGB para matplotlib
img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

# ---------------------------------------------------------
# 4. Analizar cada componente: área y bounding box
# ---------------------------------------------------------
info_componentes = []

for lab in range(1, num_labels):  # ignoramos 0 (fondo)
    mascara = (labels == lab).astype(np.uint8)  # 1 para pixeles de esa etiqueta

    # Área = número de pixeles
    area = int(np.sum(mascara))

    # Bounding box: buscamos min/max en x,y
    ys, xs = np.where(mascara == 1)
    y_min, y_max = int(np.min(ys)), int(np.max(ys))
    x_min, x_max = int(np.min(xs)), int(np.max(xs))

    info_componentes.append({
        "label": lab,
        "area_pixeles": area,
        "bbox": (x_min, y_min, x_max, y_max)
    })

    print(f"Componente {lab}:")
    print(f"  Área (pixeles blancos)        : {area}")
    print(f"  Bounding box (x_min,y_min,x_max,y_max): {x_min}, {y_min}, {x_max}, {y_max}\n")

# También podemos dibujar los bounding boxes sobre una copia RGB
img_bbox = img_color.copy()
for comp in info_componentes:
    (x_min, y_min, x_max, y_max) = comp["bbox"]
    cv2.rectangle(
        img_bbox,
        (x_min, y_min),
        (x_max, y_max),
        color=(0, 255, 255),  # amarillo en BGR
        thickness=2
    )
img_bbox_rgb = cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB)

print("Resumen:")
print(f"  Objetos detectados (sin contar fondo): {len(info_componentes)}\n")

# ---------------------------------------------------------
# 5. Mostrar resultados con matplotlib
# ---------------------------------------------------------
plt.figure(figsize=(12,6))

plt.subplot(1,3,1)
plt.imshow(img_bin, cmap="gray")
plt.title("Imagen Binaria (entrada)")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(img_color_rgb)
plt.title("Componentes Conectadas\n(etiquetas con color)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(img_bbox_rgb)
plt.title("Componentes + Bounding Box")
plt.axis("off")

plt.tight_layout()
plt.show()

# Comentarios finales:
#   • connectedComponents hace el trabajo pesado:
#     recorre la imagen y asigna una etiqueta única
#     a cada región conectada de pixeles blancos.
#
#   • info_componentes te da:
#     - cuántos objetos hay
#     - qué tan grandes son
#     - dónde están
#
#   • Esto es esencial en aplicaciones industriales:
#     inspección visual, conteo de piezas,
#     seguimiento de blobs en visión robótica,
#     identificación de defectos aislados.
