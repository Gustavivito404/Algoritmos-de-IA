# =========================================================
# 48 - TEXTURAS Y SOMBRAS (frecuencias bajas vs altas)
# ---------------------------------------------------------
# Descripción:
#   En visión por computador necesitamos distinguir entre:
#     - variaciones lentas de luz / sombra / iluminación,
#     - detalles finos repetitivos de superficie (textura).
#
#   Ejemplo clásico:
#     Una pieza metálica puede tener un cambio de brillo
#     por la luz (sombra), pero también un patrón grabado
#     o rugosidad (textura). Son fenómenos distintos.
#
#   En este script vamos a:
#     1) Generar una imagen sintética con dos zonas:
#         • Zona izquierda: iluminación suave (gradiente).
#         • Zona derecha: textura repetitiva (rejilla).
#
#     2) Combinar ambas en una sola imagen.
#
#     3) Extraer:
#         • Componente de baja frecuencia (iluminación/sombra)
#           usando desenfoque gaussiano fuerte.
#         • Componente de alta frecuencia (textura)
#           usando resta imagen - iluminación y también un kernel sharpen.
#
#   Librerías:
#     - OpenCV (cv2)
#     - NumPy
#     - Matplotlib
#
#   Nota visual:
#     Esto es base para:
#       • Inspección de defectos en materiales.
#       • Detección de desgaste vs suciedad de iluminación.
#       • Visión robótica en ambientes no controlados.
# =========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Generar zona de iluminación suave (gradiente)
# ---------------------------------------------------------
alto, ancho = 200, 400

# Gradiente horizontal suave: valores de 50 a 200
# simulando iluminación que cae sobre una superficie lisa.
x = np.linspace(50, 200, ancho // 2, dtype=np.float32)
zona_iluminacion = np.tile(x, (alto, 1))  # repetir filas

# ---------------------------------------------------------
# 2. Generar zona de textura repetitiva (rejilla)
# ---------------------------------------------------------
# Creamos patrón ajedrezado pequeño repetido.
tile_size = 10  # tamaño de bloque de la rejilla
bloque = np.zeros((tile_size, tile_size), dtype=np.float32)
bloque[:, :tile_size//2] = 230.0  # mitad clara
bloque[:, tile_size//2:] = 80.0   # mitad oscura

# Repetimos este bloque para llenar el lado derecho
rep_x = (ancho // 2) // tile_size + 1
rep_y = alto // tile_size + 1
zona_textura = np.tile(bloque, (rep_y, rep_x))
zona_textura = zona_textura[:alto, :ancho//2]  # recortamos exacto
# Nota: esta zona parece una textura fuerte, tipo metal rugoso o tela gruesa.

# ---------------------------------------------------------
# 3. Combinar las dos mitades lado a lado
# ---------------------------------------------------------
# Concatenamos horizontalmente:
# izquierda = iluminación suave
# derecha   = textura
img_sintetica = np.concatenate([zona_iluminacion, zona_textura], axis=1)

# Convertimos a uint8 para operaciones OpenCV
img_sintetica_u8 = np.clip(img_sintetica, 0, 255).astype(np.uint8)

print("Imagen sintética creada:")
print("  Mitad izquierda  -> gradiente suave (sombra / iluminación)")
print("  Mitad derecha    -> patrón repetitivo (textura)")
print(f"  Dimensiones: {img_sintetica_u8.shape} (alto, ancho)")
print(f"  Brillo promedio global: {np.mean(img_sintetica_u8):.2f}\n")

# ---------------------------------------------------------
# 4. Extraer componente de baja frecuencia (iluminación)
# ---------------------------------------------------------
# Aplicamos un blur gaussiano FUERTE (kernel grande).
# Esto promedia zonas grandes, se queda con cambios suaves.
baja_freq = cv2.GaussianBlur(img_sintetica_u8, (31, 31), sigmaX=10)

print("Extraída iluminación aproximada (baja frecuencia):")
print("  -> GaussianBlur kernel 31x31, sigma=10")
print("  Esto elimina detalles finos de textura y conserva luz/sombra.\n")

# ---------------------------------------------------------
# 5. Extraer componente de alta frecuencia (textura)
# ---------------------------------------------------------
# Alta frecuencia ≈ imagen - baja_frecuencia
alta_freq = cv2.subtract(img_sintetica_u8, baja_freq)

# También aplicamos un filtro de realce tipo sharpen para resaltar bordes finos.
kernel_sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)
realzada = cv2.filter2D(img_sintetica_u8, -1, kernel_sharpen)

print("Alta frecuencia calculada:")
print("  alta_freq = imagen - baja_freq")
print("  Esto resalta textura, rugosidad, patrones repetidos.")
print("  También se calculó una versión 'realzada' con sharpen.\n")

# ---------------------------------------------------------
# 6. Mostrar resultados
# ---------------------------------------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img_sintetica_u8, cmap="gray", vmin=0, vmax=255)
plt.title("Imagen Sintética\n(Luz suave + Textura)")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(baja_freq, cmap="gray", vmin=0, vmax=255)
plt.title("Baja Frecuencia\n(Iluminación / sombra)")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(alta_freq, cmap="gray")
plt.title("Alta Frecuencia\n(Textura aislada)")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(realzada, cmap="gray", vmin=0, vmax=255)
plt.title("Sharpen (Realce local)")
plt.axis("off")

# mapa de diferencias absolutas entre original y blur,
# útil para ver zonas granulosas vs lisas
diff_map = cv2.absdiff(img_sintetica_u8, baja_freq)
plt.subplot(2, 3, 6)
plt.imshow(diff_map, cmap="gray")
plt.title("Mapa de detalle local")
plt.axis("off")

plt.tight_layout()
plt.show()

# Comentario final:
#   - La mitad izquierda tenía cambios suaves (sombra simulada).
#     Eso aparece claramente en "Baja Frecuencia".
#
#   - La mitad derecha tenía patrón repetido.
#     Eso domina en "Alta Frecuencia" y en el "Mapa de detalle".
#
#   - Esta separación es FUNDAMENTAL en visión:
#       * control de calidad industrial (raspaduras microscópicas)
#       * inspección de superficies
#       * análisis de texturas vs reflejos
