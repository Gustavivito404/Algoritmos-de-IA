# =========================================================
# 24 - CLASIFICADOR NAÏVE BAYES (SPAM vs NO SPAM)
# ---------------------------------------------------------
# Descripción:
#   El modelo Naïve Bayes clasifica instancias según la
#   probabilidad de que pertenezcan a una clase, asumiendo
#   que las características son condicionalmente independientes.
#
#   Fórmula:
#       P(C | X) ∝ P(C) * ∏ P(x_i | C)
#
#   Ejemplo:
#       Clasificamos correos electrónicos como SPAM o NO SPAM
#       usando la presencia de ciertas palabras clave:
#           - "gratis"
#           - "oferta"
#           - "urgente"
#           - "reunión"
#           - "proyecto"
#
#   Entrenamos con un conjunto simple de correos ya clasificados,
#   y luego predecimos la clase de un nuevo correo.
#
# =========================================================

from typing import List, Dict
import math

# ---------------------------------------------------------
# 1. Conjunto de entrenamiento
# ---------------------------------------------------------
# Cada correo se representa como un diccionario de palabras (1=presente, 0=ausente)
# y su clase: 'spam' o 'no_spam'
datos_entrenamiento = [
    ({"gratis": 1, "oferta": 1, "urgente": 1, "reunión": 0, "proyecto": 0}, "spam"),
    ({"gratis": 0, "oferta": 1, "urgente": 0, "reunión": 1, "proyecto": 0}, "no_spam"),
    ({"gratis": 1, "oferta": 1, "urgente": 0, "reunión": 0, "proyecto": 0}, "spam"),
    ({"gratis": 0, "oferta": 0, "urgente": 0, "reunión": 1, "proyecto": 1}, "no_spam"),
    ({"gratis": 1, "oferta": 0, "urgente": 1, "reunión": 0, "proyecto": 0}, "spam"),
    ({"gratis": 0, "oferta": 0, "urgente": 0, "reunión": 1, "proyecto": 1}, "no_spam"),
]

# ---------------------------------------------------------
# 2. Entrenamiento del modelo Naïve Bayes
# ---------------------------------------------------------
def entrenar_naive_bayes(datos: List):
    """Calcula P(C) y P(palabra|C) para cada clase."""
    clases = {}
    total_docs = len(datos)

    # Inicialización
    for _, clase in datos:
        if clase not in clases:
            clases[clase] = {"total": 0, "palabras": {}}

    # Contar ocurrencias
    for features, clase in datos:
        clases[clase]["total"] += 1
        for palabra, presente in features.items():
            if palabra not in clases[clase]["palabras"]:
                clases[clase]["palabras"][palabra] = {"presente": 0, "ausente": 0}
            if presente:
                clases[clase]["palabras"][palabra]["presente"] += 1
            else:
                clases[clase]["palabras"][palabra]["ausente"] += 1

    # Calcular probabilidades
    modelo = {}
    for clase, info in clases.items():
        modelo[clase] = {}
        modelo[clase]["P(C)"] = info["total"] / total_docs

        modelo[clase]["P(palabra|C)"] = {}
        for palabra, conteo in info["palabras"].items():
            total = info["total"]
            # Suavizado de Laplace: evita ceros
            modelo[clase]["P(palabra|C)"][palabra] = {
                1: (conteo["presente"] + 1) / (total + 2),
                0: (conteo["ausente"] + 1) / (total + 2)
            }
    return modelo

modelo = entrenar_naive_bayes(datos_entrenamiento)

# ---------------------------------------------------------
# 3. Clasificación de un nuevo correo
# ---------------------------------------------------------
def predecir_naive_bayes(modelo: Dict, nuevo_correo: Dict):
    """Aplica la regla de Naïve Bayes para clasificar."""
    probabilidades = {}

    for clase, info in modelo.items():
        # Iniciamos con el log de P(C)
        log_prob = math.log(info["P(C)"])
        for palabra, presente in nuevo_correo.items():
            if palabra in info["P(palabra|C)"]:
                log_prob += math.log(info["P(palabra|C)"][palabra][presente])
        probabilidades[clase] = log_prob

    # Convertimos de log a probabilidad normal (opcional)
    probs_exp = {c: math.exp(v) for c, v in probabilidades.items()}
    suma = sum(probs_exp.values())
    probs_norm = {c: v / suma for c, v in probs_exp.items()}

    # Determinar clase más probable
    clase_predicha = max(probs_norm, key=probs_norm.get)
    return clase_predicha, probs_norm

# ---------------------------------------------------------
# 4. Prueba del modelo
# ---------------------------------------------------------
nuevo_correo = {"gratis": 1, "oferta": 1, "urgente": 0, "reunión": 0, "proyecto": 0}

print("==============================================")
print("TRACE CLASIFICADOR NAÏVE BAYES")
print("==============================================")
print("Correo nuevo:")
print(nuevo_correo)

prediccion, probs = predecir_naive_bayes(modelo, nuevo_correo)
print("\nProbabilidades estimadas:")
for c, p in probs.items():
    print(f"  {c}: {p:.4f}")

print("\n==============================================")
print("RESULTADO FINAL")
print("==============================================")
print(f"Clasificación final → {prediccion.upper()}")
print("\nInterpretación:")
print(" - El modelo aprende las probabilidades P(palabra|clase)")
print(" - Luego combina estas evidencias usando la regla de Bayes.")
print(" - Naïve Bayes asume independencia condicional entre palabras.")
