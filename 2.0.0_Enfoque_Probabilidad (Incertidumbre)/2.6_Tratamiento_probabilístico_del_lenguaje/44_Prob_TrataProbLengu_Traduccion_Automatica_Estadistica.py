# =========================================================
# 44 - TRADUCCIÓN AUTOMÁTICA ESTADÍSTICA (versión alineada)
# ---------------------------------------------------------
# Descripción:
#   Igual que antes, queremos estimar P(inglés | español),
#   pero ahora NO vamos a contar "todas las palabras con todas".
#
#   En su lugar vamos a alinear palabra a palabra por posición:
#       es[0] ↔ en[0]
#       es[1] ↔ en[1]
#       ...
#   Esto reduce el efecto tóxico de palabras súper frecuentes
#   como "the", que aparecían en TODAS las oraciones.
#
#   Resultado:
#     La tabla de traducción será mucho más intuitiva:
#     'niña' -> 'girl', 'come' -> 'eats', etc.
# =========================================================

from collections import defaultdict
from typing import List, Dict, Tuple

# 1. Corpus paralelo
corpus_paralelo = [
    ("la niña come comida",          "the girl eats food"),
    ("el niño come comida",          "the boy eats food"),
    ("la niña juega con la pelota",  "the girl plays with the ball"),
    ("el niño juega con la pelota",  "the boy plays with the ball"),
    ("la red neuronal aprende",      "the neural network learns"),
    ("la inteligencia artificial aprende", "the artificial intelligence learns"),
    ("el sistema busca información", "the system retrieves information"),
]

def tokenizar(txt: str) -> List[str]:
    return txt.lower().split()

pares_tok = [
    (tokenizar(es), tokenizar(en))
    for (es, en) in corpus_paralelo
]

# 2. Contar co-ocurrencias alineadas por índice
coocurrencias = defaultdict(lambda: defaultdict(float))

for esp_tokens, eng_tokens in pares_tok:
    L = min(len(esp_tokens), len(eng_tokens))
    for i in range(L):
        w_es = esp_tokens[i]
        w_en = eng_tokens[i]
        coocurrencias[w_es][w_en] += 1.0

# 3. Normalizar a probabilidades P(en|es)
tabla_traduccion: Dict[str, Dict[str, float]] = {}

for w_es, dict_en in coocurrencias.items():
    total = sum(dict_en.values())
    tabla_traduccion[w_es] = {}
    for w_en, c in dict_en.items():
        tabla_traduccion[w_es][w_en] = c / total

def mejor_traduccion(w_es: str) -> Tuple[str, float]:
    if w_es not in tabla_traduccion:
        return (w_es, 0.0)
    candidatos = tabla_traduccion[w_es]
    w_en = max(candidatos, key=lambda z: candidatos[z])
    return (w_en, candidatos[w_en])

def traducir_oracion(esp: str) -> List[Tuple[str, str, float]]:
    toks = tokenizar(esp)
    salida = []
    for w in toks:
        w_en, p = mejor_traduccion(w)
        salida.append((w, w_en, p))
    return salida

# 4. Demo
oraciones_prueba = [
    "la niña come comida",
    "el sistema aprende información",
    "la inteligencia neuronal busca comida",
    "el niño juega con la red",
]

print("==============================================")
print("TABLA DE TRADUCCIÓN APRENDIDA (Top por palabra)")
print("==============================================")
for w_es, opciones_en in tabla_traduccion.items():
    orden = sorted(opciones_en.items(), key=lambda x: x[1], reverse=True)
    top = orden[0]
    print(f"{w_es:>15s} -> {top[0]} ({top[1]:.2f})   // candidatos: {orden}")
print("")

print("==============================================")
print("TRADUCCIONES DE PRUEBA")
print("==============================================\n")

for frase in oraciones_prueba:
    traduccion = traducir_oracion(frase)
    traduccion_en = " ".join([t[1] for t in traduccion])
    print(f"Frase ES: {frase}")
    print(f"Frase EN (estimado): {traduccion_en}")
    print("Detalle palabra por palabra:")
    for (w_es, w_en, p) in traduccion:
        print(f"  {w_es:>12s} -> {w_en:<12s} P={p:.2f}")
    print("")

# ---------------------------------------------------------
# 7. Interpretación
# ---------------------------------------------------------
"""
Interpretación:
---------------
- La tabla 'tabla_traduccion' actúa como una memoria
  probabilística de traducción palabra→palabra.
- Para cada palabra en español calculamos:
      P(ingles | español)
  a partir de coocurrencias en el corpus paralelo.

- Al traducir una oración nueva:
    "la inteligencia neuronal busca comida"
  puede salir algo tipo:
    "the artificial network retrieves food"

  que es una mezcla graciosa pero semánticamente
  bastante razonable, considerando lo simple del modelo.

Limitaciones:
-------------
- No maneja orden (sujeto-verbo-objeto cambia entre idiomas).
- No maneja concordancia ("the girl"/"the boy").
- No maneja palabras nuevas que nunca vio.

Pero:
-----
Este esquema fue la base de la Traducción Automática Estadística:
   • modelos IBM 1-5,
   • alineación palabra-palabra,
   • luego modelos de frase (phrase-based SMT),
   • y después, mucho más tarde, seq2seq neuronal.

Lo que hiciste aquí es literalmente la mini-semilla
de Google Translate versión 2006-2014.
"""