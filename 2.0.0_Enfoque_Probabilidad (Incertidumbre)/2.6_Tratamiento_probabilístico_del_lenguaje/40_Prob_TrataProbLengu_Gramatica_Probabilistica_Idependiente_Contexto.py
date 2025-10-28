# =========================================================
# 40 - GRAMÁTICAS PROBABILÍSTICAS INDEPENDIENTES DEL CONTEXTO
# ---------------------------------------------------------
# Descripción:
#   Una PCFG (Probabilistic Context-Free Grammar) extiende
#   las gramáticas tradicionales (CFG) asignando probabilidades
#   a cada regla de producción.
#
#   Ejemplo de reglas:
#       S → NP VP       [1.0]
#       NP → Det N      [0.6]
#       NP → N          [0.4]
#       VP → V NP       [0.7]
#       VP → V          [0.3]
#
#   Con esto, podemos calcular la probabilidad de una oración
#   como el producto de las probabilidades de las reglas
#   usadas en su derivación.
#
#   Este script muestra un ejemplo simple de cómo una PCFG
#   genera y evalúa oraciones posibles.
# =========================================================

import random

# ---------------------------------------------------------
# 1. Definimos una gramática probabilística
# ---------------------------------------------------------
PCFG = {
    "S":  [("NP VP", 1.0)],
    "NP": [("Det N", 0.6), ("N", 0.4)],
    "VP": [("V NP", 0.7), ("V", 0.3)],
    "Det": [("el", 0.5), ("la", 0.5)],
    "N": [("gato", 0.4), ("niña", 0.3), ("comida", 0.3)],
    "V": [("come", 0.5), ("duerme", 0.5)],
}

# ---------------------------------------------------------
# 2. Función para generar oraciones aleatorias según la PCFG
# ---------------------------------------------------------
def generar(sentencia="S"):
    """Expande recursivamente los símbolos no terminales."""
    palabras = []
    for simbolo in sentencia.split():
        if simbolo in PCFG:
            # elegir una regla según probabilidad
            reglas = PCFG[simbolo]
            elecciones, probs = zip(*reglas)
            siguiente = random.choices(elecciones, weights=probs)[0]
            palabras.extend(generar(siguiente))
        else:
            palabras.append(simbolo)
    return palabras

# ---------------------------------------------------------
# 3. Calcular la probabilidad de una derivación específica
# ---------------------------------------------------------
def prob_derivacion(derivacion):
    """Calcula la probabilidad total de una secuencia de reglas."""
    prob_total = 1.0
    for regla in derivacion:
        lhs, rhs = regla
        for produccion, p in PCFG[lhs]:
            if produccion == rhs:
                prob_total *= p
                break
    return prob_total

# ---------------------------------------------------------
# 4. Ejemplo manual de derivación
# ---------------------------------------------------------
derivacion_ejemplo = [
    ("S", "NP VP"),
    ("NP", "Det N"),
    ("Det", "el"),
    ("N", "gato"),
    ("VP", "V"),
    ("V", "duerme"),
]

probabilidad = prob_derivacion(derivacion_ejemplo)

print("==============================================")
print("GRAMÁTICA PROBABILÍSTICA INDEPENDIENTE DEL CONTEXTO")
print("==============================================\n")

print("Ejemplo de derivación:")
for lhs, rhs in derivacion_ejemplo:
    print(f"  {lhs} → {rhs}")

print(f"\nProbabilidad total de la oración = {probabilidad:.5f}")

print("\nOración generada:", " ".join(generar()))
print("Otra posible oración:", " ".join(generar()))
print("Y otra:", " ".join(generar()))

# ---------------------------------------------------------
# 5. Interpretación
# ---------------------------------------------------------
"""
Interpretación:
---------------
- Cada oración generada sigue reglas sintácticas válidas.
- La probabilidad depende de cuán “comunes” son las reglas usadas.
- Este tipo de gramática fue la base de los parsers
  estadísticos previos a los transformadores modernos.
- Permiten modelar lenguaje con estructura jerárquica
  (no solo secuencial como en los n-gramas).
"""
