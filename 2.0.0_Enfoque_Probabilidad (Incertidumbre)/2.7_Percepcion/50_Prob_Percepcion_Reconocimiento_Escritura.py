# =========================================================
# 50 - RECONOCIMIENTO DE ESCRITURA (OCR Clásico con Plantillas)
# ---------------------------------------------------------
# Descripción:
#   Reconocimiento de caracteres escritos usando
#   comparación directa contra plantillas conocidas.
#
#   Idea básica (OCR clásico antes de las redes neuronales):
#     1) Tengo prototipos de letras/dígitos (A, B, C, ...),
#        cada uno como imagen binaria o en escala de grises.
#     2) Recibo una imagen desconocida con UNA letra.
#     3) Calculo qué plantilla se parece más.
#     4) Reporto esa letra como "reconocida".
#
#   En este script:
#     - Generamos imágenes sintéticas de letras usando OpenCV.
#     - Creamos un diccionario de plantillas { 'A': imgA, 'B': imgB, ... }.
#     - Comparamos una letra desconocida contra cada plantilla
#       midiendo similitud tipo correlación normalizada.
#     - Mostramos la mejor coincidencia.
#
#   Nota:
#     Esto es la raíz de OCR "clásico":
#       • segmentar cada carácter
#       • normalizarlo (mismo tamaño, centrado)
#       • compararlo contra prototipos
#
#     Hoy día se usan redes neuronales convolucionales,
#     pero la idea de "prototipo más cercano" sigue viva
#     en algunas etapas de visión industrial sencilla.
#
# =========================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# ---------------------------------------------------------
# 1. Función para generar una imagen con una letra
# ---------------------------------------------------------
def generar_letra_img(
    letra: str,
    ancho: int = 80,
    alto: int = 80,
    escala: float = 2.5,
    grosor: int = 4
) -> np.ndarray:
    """
    Genera una imagen en escala de grises con fondo negro y
    una letra blanca dibujada con OpenCV.
    - letra: carácter a dibujar (ej. 'A')
    - ancho, alto: tamaño de la imagen en pixeles
    - escala: escala del texto (fontScale en putText)
    - grosor: grosor de la línea de texto
    """
    img = np.zeros((alto, ancho), dtype=np.uint8)  # fondo negro

    # cv2.putText dibuja texto blanco (255)
    # NOTA: el origen es la esquina inferior-izquierda del texto
    cv2.putText(
        img,
        letra,
        org=(10, int(alto*0.75)),  # desplazamiento fijo razonable
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=escala,
        color=255,
        thickness=grosor,
        lineType=cv2.LINE_AA
    )
    return img

# ---------------------------------------------------------
# 2. Construimos nuestras "plantillas" conocidas
# ---------------------------------------------------------
plantillas: Dict[str, np.ndarray] = {
    "A": generar_letra_img("A"),
    "B": generar_letra_img("B"),
    "C": generar_letra_img("C"),
}

# ---------------------------------------------------------
# 3. Generamos una letra "desconocida"
# ---------------------------------------------------------
# Aquí puedes cambiar la letra para probar reconocimiento,
# por ejemplo "A", "B", "C", o incluso algo distinto como "B"
# con diferentes escala/grosor para simular escritura distinta.
letra_real = "B"
img_desconocida = generar_letra_img(letra_real, escala=2.5, grosor=5)

print("Letra desconocida generada sintéticamente:")
print(f"  Debería ser: '{letra_real}' (esto es la 'verdad oculta')\n")

# ---------------------------------------------------------
# 4. Función de similitud entre dos imágenes
# ---------------------------------------------------------
def similitud_correlacion_norm(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calcula una similitud tipo 'coseno' entre dos imágenes,
    interpretándolas como vectores.
    IMPORTANTE:
      - Ambas imágenes deben tener el MISMO tamaño.
    """
    I1 = img1.astype(np.float32).flatten()
    I2 = img2.astype(np.float32).flatten()

    num = np.sum(I1 * I2)
    den = np.sqrt(np.sum(I1**2) * np.sum(I2**2)) + 1e-9
    return num / den  # rango aprox 0..1, más alto = más parecido

# ---------------------------------------------------------
# 5. Normalizamos tamaños antes de comparar
# ---------------------------------------------------------
# Forzamos todas las plantillas a tener el mismo tamaño que la desconocida
alto_ref, ancho_ref = img_desconocida.shape

plantillas_redimensionadas: Dict[str, np.ndarray] = {}
for letra, img_tpl in plantillas.items():
    # Redimensionamos plantilla al tamaño de la imagen desconocida
    img_resized = cv2.resize(img_tpl, (ancho_ref, alto_ref), interpolation=cv2.INTER_LINEAR)
    plantillas_redimensionadas[letra] = img_resized

# ---------------------------------------------------------
# 6. Calculamos la similitud entre la desconocida y cada plantilla
# ---------------------------------------------------------
resultados_similitud: Dict[str, float] = {}
for letra, tpl_img in plantillas_redimensionadas.items():
    score = similitud_correlacion_norm(img_desconocida, tpl_img)
    resultados_similitud[letra] = score
    print(f"Similitud con '{letra}': {score:.4f}")

# Elegimos la mejor coincidencia
letra_predicha = max(resultados_similitud, key=lambda L: resultados_similitud[L])
mejor_score = resultados_similitud[letra_predicha]

print("\n==============================================")
print("RESULTADO DEL RECONOCIMIENTO")
print("==============================================")
print(f"Letra verdadera         : '{letra_real}'")
print(f"Letra predicha (OCR)    : '{letra_predicha}'")
print(f"Score de similitud      : {mejor_score:.4f}")
print("Nota: mayor score = más parecido visualmente.\n")

# ---------------------------------------------------------
# 7. Visualización con matplotlib
# ---------------------------------------------------------
plt.figure(figsize=(10,4))

# Imagen desconocida
plt.subplot(1, 4, 1)
plt.imshow(img_desconocida, cmap="gray")
plt.title(f"Desconocida\n'{letra_real}'")
plt.axis("off")

# Plantilla A
plt.subplot(1, 4, 2)
plt.imshow(plantillas_redimensionadas["A"], cmap="gray")
plt.title(f"Plantilla 'A'\nscore={resultados_similitud['A']:.2f}")
plt.axis("off")

# Plantilla B
plt.subplot(1, 4, 3)
plt.imshow(plantillas_redimensionadas["B"], cmap="gray")
plt.title(f"Plantilla 'B'\nscore={resultados_similitud['B']:.2f}")
plt.axis("off")

# Plantilla C
plt.subplot(1, 4, 4)
plt.imshow(plantillas_redimensionadas["C"], cmap="gray")
plt.title(f"Plantilla 'C'\nscore={resultados_similitud['C']:.2f}")
plt.axis("off")

plt.tight_layout()
plt.show()

# Comentarios finales:
#   • Este método es básicamente "comparación visual directa".
#   • Es MUY sensible a:
#       - rotación
#       - escala
#       - fuente distinta / estilo de escritura
#   • Pero en entornos controlados (por ejemplo, leer caracteres
#     impresos en un display de 7 segmentos, o códigos alfanuméricos
#     grabados por láser en una pieza) esto funciona sorprendentemente bien.
#
#   • Esta técnica es el puente histórico entre visión clásica
#     y OCR moderno basado en redes neuronales convolucionales (CNNs).
