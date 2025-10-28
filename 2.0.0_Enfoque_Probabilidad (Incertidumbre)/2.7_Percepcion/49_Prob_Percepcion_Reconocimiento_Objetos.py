# =========================================================
# 49 - RECONOCIMIENTO DE OBJETOS (Template Matching Clásico)
# ---------------------------------------------------------
# Descripción:
#   Este script implementa reconocimiento de objetos
#   usando "template matching": buscamos una subimagen
#   (template) dentro de una imagen más grande.
#
#   Toma:
#     - imagen_T : recorte del objeto que quiero detectar
#                  (por ejemplo, una manzana).
#     - imagen_I : escena completa donde podría aparecer
#                  esa manzana.
#
#   Objetivo:
#     Encontrar en qué parte de la imagen grande aparece
#     el patrón del template y marcarlo con un rectángulo.
#
#   Métricas usadas:
#
#   1) SSD (Sum of Squared Differences)
#      - Para cada posición (i,j), calculamos:
#           SSD(i,j) = Σ ( T(x,y) - I_sub(x,y) )^2
#        donde I_sub es la ventana de la imagen grande
#        alineada con el template.
#      - Cuanto más pequeño el SSD, más parecidas son.
#
#   2) Correlación normalizada (similaridad coseno)
#      - Para cada posición:
#           sim(i,j) = Σ [T * I_sub] /
#                      sqrt( Σ[T^2] * Σ[I_sub^2] )
#      - Cuanto más grande sim(i,j), mejor coincidencia.
#
#   Flujo:
#     • Convertimos ambas imágenes a escala de grises.
#     • Deslizamos el template por toda la imagen grande.
#     • Guardamos mapa de SSD y de similitud.
#     • Elegimos:
#           - el mínimo SSD  (mejor match por error bajo)
#           - el máximo sim  (mejor match por alta correlación)
#     • Dibujamos rectángulos en esas ubicaciones.
#
#   Visualización:
#     - Se muestra:
#         * El template
#         * La escena con el cuadro verde (SSD)
#         * La escena con el cuadro azul (correlación)
#         * Los mapas de calor de SSD y correlación
#
#   Uso típico:
#     - Reconocer un objeto conocido en una escena.
#     - Detección rígida sin rotación ni escala.
#     - Inspección industrial de piezas (¿está la pieza donde debe?).
#
#   Limitaciones:
#     - Supone que el objeto no rota ni cambia de tamaño.
#     - Muy sensible a iluminación diferente.
#     - Solo funciona bien si el template está casi idéntico
#       dentro de la imagen.
#
#   Nota:
#     Esto es reconocimiento de objetos "clásico", previo
#     a redes neuronales tipo YOLO / Faster R-CNN.
# =========================================================

import cv2
import numpy as np

# Cargar imagen y declarar variables a utilizar
imagen_T = cv2.imread(r'C:\Users\Gusta\Desktop\Inteligencia artificial\Algoritmos-de-IA\2.0.0_Enfoque_Probabilidad (Incertidumbre)\2.7_Percepcion\49_2_Template Matching_Solo Manzana.jpg')
imagen_I = cv2.imread(r'C:\Users\Gusta\Desktop\Inteligencia artificial\Algoritmos-de-IA\2.0.0_Enfoque_Probabilidad (Incertidumbre)\2.7_Percepcion\49_1_Template Matching_Manzana y paisaje.jpg')

imagen_T_gray = cv2.cvtColor(imagen_T, cv2.COLOR_BGR2GRAY)
imagen_I_gray = cv2.cvtColor(imagen_I, cv2.COLOR_BGR2GRAY)

H, W   = imagen_I_gray.shape
s1, s2 = imagen_T_gray.shape

I = imagen_I_gray.astype(np.float32)
T = imagen_T_gray.astype(np.float32)

# Pequeña constante
eps = 1e-9

# Mapa de salida
TM_Diferencias = np.zeros((H-s1+1, W-s2+1), dtype=np.float32)
TM_Similitudes = np.zeros((H-s1+1, W-s2+1), dtype=np.float32)

# Template matching basado en diferencia al cuadrado (SSD)
for i in range(H-s1+1):
    for j in range(W-s2+1):
        I_sub = I[i:i+s1, j:j+s2]
        diff = T - I_sub
        TM_Diferencias[i, j] = np.sum(diff ** 2)

# Template matching basado en similitud (correlación normalizada)
for i in range(H-s1+1):
    for j in range(W-s2+1):
        I_sub = I[i:i+s1, j:j+s2]
        sim = np.sum(T * I_sub) / (np.sqrt((np.sum(T ** 2) * np.sum(I_sub ** 2) + eps)))
        TM_Similitudes[i, j] = sim

# Mejor coincidencia para SSD
min_val = np.min(TM_Diferencias)
min_loc = np.unravel_index(np.argmin(TM_Diferencias), TM_Diferencias.shape)
y0, x0 = int(min_loc[0]), int(min_loc[1])

# Mejor coincidencia para correlación normalizada
max_val = np.max(TM_Similitudes)
max_loc = np.unravel_index(np.argmax(TM_Similitudes), TM_Similitudes.shape)
y1, x1 = int(max_loc[0]), int(max_loc[1])

# Dibujar rectángulo sobre la mejor coincidencia
img_color_SSD = imagen_I.copy()
cv2.rectangle(img_color_SSD, (x0, y0), (x0 + s2, y0 + s1), (0, 255, 0), 2)

# Dibujar rectángulo sobre la mejor coincidencia
img_color_corr = imagen_I.copy()
cv2.rectangle(img_color_corr, (x1, y1), (x1 + s2, y1 + s1), (255, 0, 0), 2)

# Mostrar resultados
TM_norm_Diferencias = cv2.normalize(TM_Diferencias, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
TM_norm_Similitudes = cv2.normalize(TM_Similitudes, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow('Template', imagen_T)
cv2.imshow('Resultado SSD', img_color_SSD)
cv2.imshow('Resultado Correlacion', img_color_corr)
cv2.imshow('Mapa de error SSD', cv2.applyColorMap(TM_norm_Diferencias, cv2.COLORMAP_JET))
cv2.imshow('Mapa de error Correlacion', cv2.applyColorMap(TM_norm_Similitudes, cv2.COLORMAP_JET))

print("Error mínimo:", min_val)
print("Coordenadas de coincidencia:", (x0, y0))

print("Similitud máxima:", max_val)
print("Coordenadas de coincidencia:", (x1, y1))

cv2.waitKey(0)
cv2.destroyAllWindows()