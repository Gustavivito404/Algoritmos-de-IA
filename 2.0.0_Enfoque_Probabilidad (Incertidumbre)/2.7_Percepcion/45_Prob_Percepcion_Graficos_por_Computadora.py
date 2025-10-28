# =========================================================
# 45 - GRÁFICOS POR COMPUTADOR (primitivas básicas con OpenCV)
# ---------------------------------------------------------
# Descripción:
#   En visión por computador muchas veces necesitamos generar
#   imágenes sintéticas (para pruebas, simulación, debugging
#   de algoritmos de percepción).
#
#   Aquí vamos a:
#     1) Crear una imagen en negro como "lienzo".
#     2) Dibujar primitivas geométricas:
#         - Línea
#         - Rectángulo
#         - Círculo
#         - Texto
#     3) Visualizar el resultado con matplotlib.
#
#   Esto es la base de:
#     • Renderizado sintético de escenas
#     • Datasets artificiales
#     • Etiquetado visual para debugging (por ej. dibujar bounding boxes)
#
#   Nota técnica:
#     - OpenCV trabaja en BGR (azul, verde, rojo)
#     - Matplotlib asume RGB
#     - Vamos a convertir antes de mostrar
#
# Ejemplo visual ASCII (coordenadas aproximadas):
#
#   (0,0)  ┌───────────────────────────► x
#         │   [RECTÁNGULO   ]   O CÍRCULO
#         │
#         ▼
#         y
#
#   Todo se basa en coordenadas de imagen:
#   (columna=x, fila=y)
#
# =========================================================

import numpy as np
import cv2
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Crear lienzo negro
# ---------------------------------------------------------
# Creamos una imagen de 400x600 píxeles, 3 canales (color)
# dtype=uint8 significa: valores 0..255 por canal
alto = 400
ancho = 600
lienzo = np.zeros((alto, ancho, 3), dtype=np.uint8)

print("Lienzo creado:")
print(f"  Dimensiones: {lienzo.shape}  (alto, ancho, canales)")
print(f"  Tipo de dato: {lienzo.dtype}")
print("  Todos los pixeles iniciales son negros (0,0,0)\n")

# ---------------------------------------------------------
# 2. Dibujar una línea
# ---------------------------------------------------------
# cv2.line(img, pt1, pt2, color_bgr, thickness)
pt1 = (50, 50)    # (x1, y1)
pt2 = (550, 50)   # (x2, y2)
color_linea = (255, 0, 0)  # BGR -> azul puro
cv2.line(lienzo, pt1, pt2, color_linea, thickness=3)

print("Dibujamos una LÍNEA azul:")
print(f"  Desde {pt1} hasta {pt2}")
print(f"  Grosor = 3 px")
print(f"  Color BGR={color_linea}\n")

# ---------------------------------------------------------
# 3. Dibujar un rectángulo
# ---------------------------------------------------------
# cv2.rectangle(img, pt1_sup_izq, pt2_inf_der, color_bgr, thickness)
# thickness = -1  -> relleno sólido
rect_sup_izq = (100, 100)
rect_inf_der = (250, 250)
color_rect = (0, 255, 0)  # verde en BGR
cv2.rectangle(lienzo, rect_sup_izq, rect_inf_der, color_rect, thickness=-1)

print("Dibujamos un RECTÁNGULO verde relleno:")
print(f"  Sup-Izq = {rect_sup_izq}")
print(f"  Inf-Der = {rect_inf_der}")
print(f"  Color BGR={color_rect}")
print("  thickness=-1 => relleno\n")

# ---------------------------------------------------------
# 4. Dibujar un círculo
# ---------------------------------------------------------
# cv2.circle(img, centro, radio, color_bgr, thickness)
centro_circulo = (400, 200)
radio_circulo = 60
color_circulo = (0, 0, 255)  # rojo en BGR
cv2.circle(lienzo, centro_circulo, radio_circulo, color_circulo, thickness=4)

print("Dibujamos un CÍRCULO rojo (solo borde):")
print(f"  Centro = {centro_circulo}")
print(f"  Radio  = {radio_circulo}")
print(f"  Grosor = 4 px")
print(f"  Color BGR={color_circulo}\n")

# ---------------------------------------------------------
# 5. Dibujar texto
# ---------------------------------------------------------
# cv2.putText(img, texto, origen, font, escala, color_bgr, grosor, lineType)
texto = "Wololo!"
origen_texto = (50, 350)  # esquina inferior izquierda del texto
color_texto = (255, 255, 255)  # blanco
cv2.putText(
    lienzo,
    texto,
    origen_texto,
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=1.2,
    color=color_texto,
    thickness=2,
    lineType=cv2.LINE_AA
)

print("Escribimos TEXTO en blanco:")
print(f"  Texto      = '{texto}'")
print(f"  Posición   = {origen_texto} (esquina inferior izq. del texto)")
print(f"  Escala     = 1.2")
print(f"  Grosor     = 2 px")
print(f"  Color BGR  = {color_texto}")
print("  Fuente     = FONT_HERSHEY_SIMPLEX\n")

# ---------------------------------------------------------
# 6. Mostrar la imagen final (convertimos BGR -> RGB)
# ---------------------------------------------------------
lienzo_rgb = cv2.cvtColor(lienzo, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8,4))
plt.imshow(lienzo_rgb)
plt.title("Primitivas gráficas básicas dibujadas con OpenCV")
plt.axis("off")
plt.show()

# Comentario:
#   En percepción por computadora vas a hacer esto MUCHO:
#   - Dibujar bounding boxes alrededor de objetos detectados
#   - Marcar puntos clave (keypoints)
#   - Visualizar rutas de robots / drones
#   - Depurar segmentaciones y máscaras
#
#   Aquí ya tienes las funciones base para eso:
#   line(), rectangle(), circle(), putText()
