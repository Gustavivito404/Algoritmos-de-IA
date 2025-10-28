# =========================================================
# 43 - EXTRACCIÓN DE INFORMACIÓN (SVO + metadatos simples)
# ---------------------------------------------------------
# Descripción:
#   La Extracción de Información (IE) convierte texto libre
#   en datos estructurados. Ejemplo típico:
#
#      Texto: "La empresa Alpha compró la startup Beta en 2024."
#      IE:
#         Sujeto = "La empresa Alpha"
#         Verbo  = "compró"
#         Objeto = "la startup Beta"
#         Tiempo = "2024"
#
#   En este script:
#     1) Definimos un pequeño conjunto de oraciones en español.
#     2) Intentamos extraer (SUJETO, VERBO, OBJETO) usando
#        reglas muy simples basadas en el orden de palabras.
#     3) Intentamos extraer también TIEMPO y LUGAR si aparecen.
#
#   Nota:
#     Esto es una versión mini de lo que hacen los sistemas
#     de IE en periodismo automatizado, ciberseguridad,
#     análisis de reportes, etc.
#
#   Limitación:
#     Aquí NO usamos un etiquetador sintáctico real,
#     sólo heurísticas. Es didáctico.
# =========================================================

import re
from typing import List, Dict

# ---------------------------------------------------------
# 1. Texto de entrada (noticias / reportes simulados)
# ---------------------------------------------------------
documento = [
    "La empresa Alpha compró la startup Beta en 2024.",
    "El laboratorio Sigma desarrolló un nuevo sensor óptico en Ciudad Delta.",
    "La IA del Instituto Omega analizó millones de imágenes médicas.",
    "El dron Hermes inspeccionó la planta de energía durante la noche.",
    "Tecnorobotics presentó su robot Atlas en la feria de robótica 2025.",
]

# ---------------------------------------------------------
# 2. Listas de verbos de acción comunes (simplificación)
# ---------------------------------------------------------
VERBOS_ACCION = [
    "compró", "desarrolló", "analizó", "inspeccionó", "presentó",
    "adquirió", "creó", "construyó", "lanzó", "probó"
]

# ---------------------------------------------------------
# 3. Detectores simples de tiempo y lugar (reglas débiles)
# ---------------------------------------------------------
def extraer_tiempo(oracion: str) -> str:
    """
    Busca patrones tipo año (2024, 2025) o palabras como 'durante la noche'
    """
    # Busca año tipo 20xx
    m = re.search(r"\b(20[0-9]{2})\b", oracion)
    if m:
        return m.group(1)

    # Busca frases temporales comunes
    m2 = re.search(r"(durante la noche|ayer|hoy|esta mañana|anoche)", oracion, re.IGNORECASE)
    if m2:
        return m2.group(1)

    return ""

def extraer_lugar(oracion: str) -> str:
    """
    Busca patrones tipo 'en Ciudad Delta', 'en la planta de energía', 'en la feria de robótica 2025'
    Estrategia: agarro 'en ...' hasta final o hasta punto.
    """
    m = re.search(r"\ben\s+([A-ZÁÉÍÓÚÑ][^\.]+)", oracion)
    # Ejemplo: "en Ciudad Delta", "en la feria de robótica 2025"
    if m:
        return m.group(1).strip()
    return ""

# ---------------------------------------------------------
# 4. Heurística Sujeto-Verbo-Objeto (SVO)
# ---------------------------------------------------------
def extraer_svo(oracion: str) -> Dict[str,str]:
    """
    Intenta extraer:
      - SUJETO: palabras antes del verbo
      - VERBO: verbo en lista VERBOS_ACCION
      - OBJETO: palabras después del verbo hasta 'en ...' o final
    """
    tokens = oracion.replace(".", "").split()

    # localizar el primer verbo conocido en la oración
    verbo_idx = -1
    verbo_encontrado = ""
    for i, tok in enumerate(tokens):
        tnorm = tok.lower()
        if tnorm in VERBOS_ACCION:
            verbo_idx = i
            verbo_encontrado = tok
            break

    if verbo_idx == -1:
        # no encontramos verbo de acción conocido
        return {
            "sujeto": "",
            "verbo": "",
            "objeto": ""
        }

    # SUJETO = todo antes del verbo
    sujeto = " ".join(tokens[:verbo_idx])

    # OBJETO = todo después del verbo hasta (opcionalmente) una preposición 'en ...'
    resto = tokens[verbo_idx+1:]

    objeto_tokens = []
    for t in resto:
        if t.lower() == "en":  # cortamos objeto al llegar a "en"
            break
        objeto_tokens.append(t)

    objeto = " ".join(objeto_tokens)

    return {
        "sujeto": sujeto.strip(),
        "verbo": verbo_encontrado.strip(),
        "objeto": objeto.strip()
    }

# ---------------------------------------------------------
# 5. Aplicar extracción a cada oración
# ---------------------------------------------------------
print("==============================================")
print("EXTRACCIÓN DE INFORMACIÓN")
print("==============================================\n")

resultados = []
for oracion in documento:
    svo = extraer_svo(oracion)
    tiempo = extraer_tiempo(oracion)
    lugar = extraer_lugar(oracion)

    resultados.append({
        "oracion": oracion,
        "sujeto": svo["sujeto"],
        "verbo": svo["verbo"],
        "objeto": svo["objeto"],
        "tiempo": tiempo,
        "lugar": lugar,
    })

# ---------------------------------------------------------
# 6. Mostramos resultados tipo tabla legible
# ---------------------------------------------------------
for r in resultados:
    print("Oración:", r["oracion"])
    print(f"  SUJETO : {r['sujeto']}")
    print(f"  VERBO  : {r['verbo']}")
    print(f"  OBJETO : {r['objeto']}")
    print(f"  TIEMPO : {r['tiempo'] if r['tiempo'] else '(no detectado)'}")
    print(f"  LUGAR  : {r['lugar'] if r['lugar'] else '(no detectado)'}")
    print("")

# ---------------------------------------------------------
# 7. Interpretación
# ---------------------------------------------------------
"""
Interpretación:
---------------
Esto hace algo muy parecido a los primeros sistemas
de minería de texto:

- Toma una oración.
- Encuentra SUJETO, VERBO, OBJETO (relación de tipo "X hizo Y a Z").
- Extrae contexto adicional como TIEMPO y LUGAR.
- Devuelve una ficha semiestructurada.

Esto se usaba (y todavía se usa) para:
  • reportes automáticos,
  • análisis de noticias,
  • vigilancia de incidentes industriales,
  • inteligencia competitiva (quién compró a quién).

En la práctica moderna, esto se haría con modelos de etiquetado
secuencial (NER, SRL, dependencias sintácticas), pero la idea
es la misma: destilar HECHOS de TEXTO.
"""
