# =========================================================
# 41 - GRAMÁTICAS PROBABILÍSTICAS LEXICALIZADAS
# ---------------------------------------------------------
# Descripción:
#   Las gramáticas probabilísticas lexicalizadas (Lexicalized PCFG)
#   extienden las PCFG asignando no solo una probabilidad a cada
#   regla de producción sintáctica, sino también un "núcleo léxico"
#   (palabra cabeza) asociado a cada constituyente.
#
#   Ejemplo de nodo lexicalizado:
#       NP(niña), VP(come), NP(comida)
#
#   Eso permite capturar dependencias semánticas reales como:
#       "la niña come comida" tiene alta probabilidad,
#       "la niña juega comida" debería tener baja probabilidad.
#
#   En este script:
#     - Definimos una mini gramática lexicalizada.
#     - Mostramos cómo generar oraciones aleatorias.
#     - Calculamos la probabilidad de una derivación completa.
#
#   Nota:
#     Este tipo de modelo fue muy importante en el parsing
#     estadístico antes del deep learning.
# =========================================================

import random

# ---------------------------------------------------------
# 1. Gramática probabilística lexicalizada
# ---------------------------------------------------------
Lexical_PCFG = {
    # Oración completa: S
    "S": [
        ("NP(niña) VP(come)", 0.6),
        ("NP(niño) VP(juega)", 0.4)
    ],

    # Sintagmas nominales con núcleo "niña" o "niño"
    "NP(niña)": [
        ("Det N(niña)", 0.8),
        ("N(niña)", 0.2)
    ],
    "NP(niño)": [
        ("Det N(niño)", 0.7),
        ("N(niño)", 0.3)
    ],

    # Sintagmas nominales con núcleo "comida" o "pelota"
    # (esto estaba faltando y causaba KeyError)
    "NP(comida)": [
        ("Det N(comida)", 1.0)
    ],
    "NP(pelota)": [
        ("Det N(pelota)", 1.0)
    ],

    # Sintagmas verbales: dependen del verbo principal
    "VP(come)": [
        ("V(come) NP(comida)", 0.7),
        ("V(come)", 0.3)
    ],
    "VP(juega)": [
        ("V(juega) NP(pelota)", 0.6),
        ("V(juega)", 0.4)
    ],

    # Determinantes (artículos)
    "Det": [
        ("el", 0.5),
        ("la", 0.5)
    ],

    # Nombres lexicalizados
    "N(niña)": [
        ("niña", 1.0)
    ],
    "N(niño)": [
        ("niño", 1.0)
    ],
    "N(comida)": [
        ("comida", 1.0)
    ],
    "N(pelota)": [
        ("pelota", 1.0)
    ],

    # Verbos lexicalizados
    "V(come)": [
        ("come", 1.0)
    ],
    "V(juega)": [
        ("juega", 1.0)
    ],
}

# ---------------------------------------------------------
# 2. Generador de oraciones lexicalizadas
# ---------------------------------------------------------
def generar_lexicalizado(sentencia="S"):
    """
    Expande recursivamente símbolos (S, NP(niña), VP(come), etc.)
    hasta llegar solo a palabras ('la', 'niña', 'come', ...).

    Mecanismo:
    - Si el símbolo es no terminal (está en Lexical_PCFG),
      elegimos una regla con probabilidad dada.
    - Si el símbolo ya es palabra (por ejemplo 'niña'),
      la devolvemos.
    """
    palabras = []
    for simbolo in sentencia.split():
        if simbolo in Lexical_PCFG:
            reglas = Lexical_PCFG[simbolo]            # lista [(expansión, prob), ...]
            producciones, probs = zip(*reglas)        # separamos RHS y probabilidades
            elegido = random.choices(producciones, weights=probs)[0]
            palabras.extend(generar_lexicalizado(elegido))
        else:
            # simbolo ya es palabra terminal
            palabras.append(simbolo)
    return palabras

# ---------------------------------------------------------
# 3. Calcular probabilidad de una derivación específica
# ---------------------------------------------------------
def prob_derivacion_lexicalizada(derivacion):
    """
    derivacion es una lista de tuplas (LHS, RHS)
    donde:
        LHS es el símbolo izquierdo (ej. "NP(niña)")
        RHS es la expansión usada (ej. "Det N(niña)")

    La prob_total es el producto de las probabilidades
    de cada regla usada.
    """
    prob_total = 1.0
    for lhs, rhs in derivacion:
        if lhs not in Lexical_PCFG:
            # Si lhs es terminal puro (como "Det"->"la"),
            # no hay prob en este nivel? OJO:
            # Para terminales como "Det" sí hay prob,
            # así que también debemos manejar ese caso aquí.
            # Si lhs no está en el diccionario, significa
            # que esa parte no es una regla expandible conocida,
            # así que la ignoramos en el producto.
            continue

        encontrado = False
        for produccion, p in Lexical_PCFG[lhs]:
            if produccion == rhs:
                prob_total *= p
                encontrado = True
                break

        # Si no encontramos la regla exacta (lhs -> rhs),
        # podría ser porque rhs ya es palabra final directa.
        # Ejemplo:
        #    ("N(niña)", "niña")
        # También debemos cubrir ese caso:
        if not encontrado:
            for produccion, p in Lexical_PCFG[lhs]:
                if produccion == rhs:
                    prob_total *= p
                    encontrado = True
                    break

        # Si sigue sin encontrarse, lo dejamos tal cual.
        # (En una implementación formal deberíamos lanzar error.)

    return prob_total

# ---------------------------------------------------------
# 4. Ejemplo de derivación paso a paso
# ---------------------------------------------------------
# Esta derivación corresponde a la oración:
# "la niña come la comida"
#
# Árbol (simplificado):
#   S
#    ├─ NP(niña)
#    │    ├─ Det → la
#    │    └─ N(niña) → niña
#    └─ VP(come)
#         ├─ V(come) → come
#         └─ NP(comida)
#              ├─ Det → la
#              └─ N(comida) → comida

derivacion_ejemplo = [
    ("S", "NP(niña) VP(come)"),
    ("NP(niña)", "Det N(niña)"),
    ("Det", "la"),
    ("N(niña)", "niña"),
    ("VP(come)", "V(come) NP(comida)"),
    ("V(come)", "come"),
    ("NP(comida)", "Det N(comida)"),
    ("Det", "la"),
    ("N(comida)", "comida"),
]

probabilidad = prob_derivacion_lexicalizada(derivacion_ejemplo)

# ---------------------------------------------------------
# 5. Resultados
# ---------------------------------------------------------
print("==============================================")
print("GRAMÁTICAS PROBABILÍSTICAS LEXICALIZADAS")
print("==============================================\n")

print("Derivación lexicalizada usada:")
for lhs, rhs in derivacion_ejemplo:
    print(f"  {lhs} → {rhs}")

print(f"\nProbabilidad total (aprox): {probabilidad:.6f}\n")

print("Ejemplos de oraciones generadas aleatoriamente:")
for _ in range(3):
    print(" -", " ".join(generar_lexicalizado()))

# ---------------------------------------------------------
# 6. Interpretación rápida
# ---------------------------------------------------------
"""
Interpretación:
---------------
- Aquí no solo decimos "S → NP VP", sino "S → NP(niña) VP(come)".
  Eso carga información léxica ("niña", "come") en los nodos.

- Esta lexicalización permite modelar:
    • quién hace la acción (sujeto probable para 'come')
    • sobre qué recae la acción ("comida" encaja con "come")

- En PLN esto mejora mucho la capacidad del parser para
  preferir oraciones gramaticales y semánticamente coherentes.

- Este tipo de enfoque fue clave antes de los modelos
  neuronales grandes, y todavía es útil para análisis
  sintáctico probabilístico interpretable.
"""
