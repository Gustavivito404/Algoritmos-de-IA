# =========================================================
# 39 - MODELO PROBABILÍSTICO DEL LENGUAJE: CORPUS
# ---------------------------------------------------------
# Descripción:
#   Un *modelo probabilístico del lenguaje* estima la
#   probabilidad de una secuencia de palabras:
#
#       P(w1, w2, w3, ..., wn)
#
#   Pero calcular eso directamente es imposible
#   (hay infinitas combinaciones), así que se usa la
#   aproximación de *n-gramas*:
#
#       P(w1, w2, ..., wn) ≈ Π P(w_i | w_{i-1}, ..., w_{i-n+1})
#
#   En este ejemplo:
#     - Construimos un corpus pequeño de frases en español.
#     - Calculamos las probabilidades unigramas (1 palabra)
#       y bigramas (pares de palabras).
#     - Estimamos la probabilidad de una frase nueva
#       usando estas frecuencias.
#
#   Este es el mismo principio detrás de los primeros
#   modelos de lenguaje estadísticos (antes del Deep Learning).
# =========================================================

from collections import defaultdict
import math

# ---------------------------------------------------------
# 1. Definimos un corpus pequeño de entrenamiento
# ---------------------------------------------------------
corpus = [
    "yo amo programar",
    "yo amo comer",
    "yo estudio inteligencia artificial",
    "tu estudias programacion",
    "tu amas programar mucho",
    "ella estudia matematicas",
    "ella ama bailar",
]

# Preprocesamos: minúsculas y tokenización
sentencias = [s.lower().split() for s in corpus]

# ---------------------------------------------------------
# 2. Contamos frecuencias de unigramas y bigramas
# ---------------------------------------------------------
unigramas = defaultdict(int)
bigramas = defaultdict(int)

for oracion in sentencias:
    for i, palabra in enumerate(oracion):
        unigramas[palabra] += 1
        if i > 0:
            par = (oracion[i-1], palabra)
            bigramas[par] += 1

# Total de palabras
total_unigramas = sum(unigramas.values())

# ---------------------------------------------------------
# 3. Calculamos probabilidades P(w_i) y P(w_i | w_{i-1})
# ---------------------------------------------------------
def P_unigrama(palabra):
    return unigramas[palabra] / total_unigramas

def P_bigrama(palabra_actual, palabra_anterior):
    if unigramas[palabra_anterior] == 0:
        return 0
    return bigramas[(palabra_anterior, palabra_actual)] / unigramas[palabra_anterior]

# ---------------------------------------------------------
# 4. Evaluamos la probabilidad de una frase nueva
# ---------------------------------------------------------
def probabilidad_frase_bigramas(frase):
    """Calcula la probabilidad total P(w1,...,wn) con modelo de bigramas."""
    palabras = frase.lower().split()
    prob_total = 1
    for i in range(1, len(palabras)):
        p = P_bigrama(palabras[i], palabras[i-1])
        if p == 0:
            p = 1e-6  # suavizado (evitar cero)
        prob_total *= p
    return prob_total

# Frases para probar
frase1 = "yo amo programar"
frase2 = "ella ama programar"
frase3 = "tu amas bailar"

# ---------------------------------------------------------
# 5. Resultados
# ---------------------------------------------------------
print("==============================================")
print("CORPUS Y MODELO PROBABILÍSTICO DEL LENGUAJE")
print("==============================================\n")

print("Unigramas (frecuencias):")
for k, v in unigramas.items():
    print(f"{k:>12s} : {v}")
print("\n")

print("Bigramas (frecuencias):")
for k, v in bigramas.items():
    print(f"{k} : {v}")
print("\n")

print("==============================================")
print("PROBABILIDAD DE FRASES (Modelo de Bigramas)")
print("==============================================")
for frase in [frase1, frase2, frase3]:
    p = probabilidad_frase_bigramas(frase)
    logp = math.log(p)
    print(f"'{frase}' -> P = {p:.6f}   log(P) = {logp:.2f}")

# ---------------------------------------------------------
# 6. Interpretación
# ---------------------------------------------------------
"""
Interpretación:
---------------
- La probabilidad más alta corresponderá a las frases
  que el modelo “vio” en el corpus (por ejemplo, “yo amo programar”).
- Frases nuevas pero similares (como “ella ama programar”)
  tendrán menor probabilidad pero no cero (gracias al suavizado).
- Esto demuestra cómo el lenguaje puede representarse
  como un sistema probabilístico de dependencias locales.
"""
