# =========================================================
# 17 - FILTRADO, PREDICCIÓN, SUAVIZADO Y EXPLICACIÓN
# ---------------------------------------------------------
# Descripción:
#   Se extiende el proceso de Markov del clima agregando
#   un sensor que detecta si el suelo está mojado o seco.
#
#   Observaciones:
#       E_t ∈ {Mojado, Seco}
#
#   Objetivo:
#       Usar las observaciones para estimar la probabilidad
#       de que esté lloviendo en cada momento.
#
#   Mostramos paso a paso:
#       - Filtrado (creencia actualizada día a día)
#       - Predicción (creencia futura sin evidencia)
#       - Suavizado (reajuste posterior)
#       - Explicación (estado más probable)
# =========================================================

import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Modelo de transición (igual al anterior)
# ---------------------------------------------------------
P_transicion = {
    "Lluvia":   {"Lluvia": 0.7, "Soleado": 0.3},
    "Soleado":  {"Lluvia": 0.4, "Soleado": 0.6}
}

# ---------------------------------------------------------
# 2. Modelo sensor (evidencia)
# ---------------------------------------------------------
# P(Evidencia | Estado)
# Ejemplo:
#   Si llueve, el suelo tiene alta probabilidad de estar mojado (0.9).
#   Si está soleado, puede estar seco (0.8) o mojado por otras causas (0.2).
P_sensor = {
    "Lluvia":   {"Mojado": 0.9, "Seco": 0.1},
    "Soleado":  {"Mojado": 0.2, "Seco": 0.8}
}

# ---------------------------------------------------------
# 3. Secuencia de observaciones (evidencias)
# ---------------------------------------------------------
# Suponemos que observamos durante 5 días:
# "Mojado, Mojado, Seco, Mojado, Mojado"
observaciones = ["Mojado", "Mojado", "Seco", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado"]

# ---------------------------------------------------------
# 4. Estado inicial (creencia previa)
# ---------------------------------------------------------
creencia = {"Lluvia": 0.5, "Soleado": 0.5}
historial_lluvia = [creencia["Lluvia"]]

# ---------------------------------------------------------
# 5. Funciones auxiliares
# ---------------------------------------------------------
def normalizar(d):
    """Normaliza un diccionario de probabilidades."""
    total = sum(d.values())
    for k in d:
        d[k] /= total
    return d

def filtrar(creencia, evidencia):
    """Actualiza la creencia con base en la evidencia observada."""
    prediccion = {}
    # Paso 1: Predicción (transición)
    for x_next in ["Lluvia", "Soleado"]:
        prediccion[x_next] = (
            creencia["Lluvia"]  * P_transicion["Lluvia"][x_next] +
            creencia["Soleado"] * P_transicion["Soleado"][x_next]
        )

    # Paso 2: Actualización (incorporar evidencia)
    for x in prediccion:
        prediccion[x] *= P_sensor[x][evidencia]

    # Paso 3: Normalización
    return normalizar(prediccion)

# ---------------------------------------------------------
# 6. Aplicamos el filtrado día a día
# ---------------------------------------------------------
print("==============================================")
print("TRACE FILTRADO (día a día)")
print("==============================================")
for t, e in enumerate(observaciones, start=1):
    creencia = filtrar(creencia, e)
    historial_lluvia.append(creencia["Lluvia"])
    print(f"Día {t:2d} | Evidencia: {e:<6} | P(Lluvia)={creencia['Lluvia']:.3f} | P(Soleado)={creencia['Soleado']:.3f}")

# ---------------------------------------------------------
# 7. Predicción (qué pasa sin nueva evidencia)
# ---------------------------------------------------------
prediccion_futura = {
    "Lluvia": (
        creencia["Lluvia"]  * P_transicion["Lluvia"]["Lluvia"] +
        creencia["Soleado"] * P_transicion["Soleado"]["Lluvia"]
    ),
    "Soleado": (
        creencia["Lluvia"]  * P_transicion["Lluvia"]["Soleado"] +
        creencia["Soleado"] * P_transicion["Soleado"]["Soleado"]
    ),
}
print("\n==============================================")
print("PREDICCIÓN FUTURA (+1 día sin evidencia)")
print("==============================================")
print(f"P(Lluvia mañana) ≈ {prediccion_futura['Lluvia']:.3f}")
print(f"P(Soleado mañana) ≈ {prediccion_futura['Soleado']:.3f}")

# ---------------------------------------------------------
# 8. Visualización
# ---------------------------------------------------------
plt.plot(range(len(historial_lluvia)), historial_lluvia, marker="o", color="royalblue")
plt.title("Filtrado Bayesiano: P(Lluvia | evidencias)")
plt.xlabel("Día")
plt.ylabel("Probabilidad de lluvia")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Comentario final:
# - Cada vez que observamos "Mojado", la probabilidad de lluvia sube.
# - Cuando observamos "Seco", la probabilidad baja.
# - La curva muestra cómo el filtrado ajusta dinámicamente la creencia
#   a medida que llegan nuevas observaciones.
