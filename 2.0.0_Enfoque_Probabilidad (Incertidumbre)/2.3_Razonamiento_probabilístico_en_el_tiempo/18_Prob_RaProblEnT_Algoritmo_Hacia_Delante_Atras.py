# =========================================================
# 18 - ALGORITMO HACIA DELANTE-ATRÁS
# ---------------------------------------------------------
# Descripción:
#   Este script aplica el algoritmo forward-backward
#   (hacia adelante y hacia atrás) para el modelo del clima.
#
#   Permite calcular la probabilidad P(X_t | e_1:T),
#   es decir, la creencia del estado en cada día t
#   usando toda la evidencia disponible.
# =========================================================

import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Modelo de transición
# ---------------------------------------------------------
P_transicion = {
    "Lluvia":   {"Lluvia": 0.7, "Soleado": 0.3},
    "Soleado":  {"Lluvia": 0.4, "Soleado": 0.6}
}

# ---------------------------------------------------------
# 2. Modelo sensor (P(Evidencia | Estado))
# ---------------------------------------------------------
P_sensor = {
    "Lluvia":   {"Mojado": 0.9, "Seco": 0.1},
    "Soleado":  {"Mojado": 0.2, "Seco": 0.8}
}

# ---------------------------------------------------------
# 3. Evidencias observadas
# ---------------------------------------------------------
observaciones = ["Mojado", "Mojado", "Seco", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado", "Mojado"]

# ---------------------------------------------------------
# 4. Estado inicial
# ---------------------------------------------------------
creencia_inicial = {"Lluvia": 0.5, "Soleado": 0.5}
estados = ["Lluvia", "Soleado"]

# ---------------------------------------------------------
# 5. Normalización
# ---------------------------------------------------------
def normalizar(d):
    total = sum(d.values())
    for k in d:
        d[k] /= total
    return d

# ---------------------------------------------------------
# 6. Paso hacia adelante (forward)
# ---------------------------------------------------------
forward = [creencia_inicial]

for e in observaciones:
    nuevo = {}
    for x in estados:
        # Predicción desde ambos estados anteriores
        pred = sum(forward[-1][x_prev] * P_transicion[x_prev][x] for x_prev in estados)
        # Actualización con la nueva evidencia
        nuevo[x] = P_sensor[x][e] * pred
    forward.append(normalizar(nuevo))

# ---------------------------------------------------------
# 7. Paso hacia atrás (backward)
# ---------------------------------------------------------
backward = [{x: 1.0 for x in estados}]  # al final, β_T(x) = 1

for t in range(len(observaciones) - 1, -1, -1):
    nuevo = {}
    for x in estados:
        nuevo[x] = sum(
            P_transicion[x][x_next] * P_sensor[x_next][observaciones[t]] * backward[0][x_next]
            for x_next in estados
        )
    backward.insert(0, normalizar(nuevo))

# ---------------------------------------------------------
# 8. Suavizado (combinación de α y β)
# ---------------------------------------------------------
suavizado = []
for t in range(1, len(forward)):
    suav = {}
    for x in estados:
        suav[x] = forward[t][x] * backward[t][x]
    suavizado.append(normalizar(suav))

# ---------------------------------------------------------
# 9. Resultados paso a paso
# ---------------------------------------------------------
print("==============================================")
print("TRACE SUAVIZADO (Forward-Backward)")
print("==============================================")
for i, s in enumerate(suavizado, start=1):
    print(f"Día {i:2d} | P(Lluvia | e₁:T)={s['Lluvia']:.3f} | P(Soleado)={s['Soleado']:.3f}")

# ---------------------------------------------------------
# 10. Visualización
# ---------------------------------------------------------
filtrado = [f["Lluvia"] for f in forward[1:]]
suavizado_lluvia = [s["Lluvia"] for s in suavizado]

plt.plot(range(1, len(observaciones) + 1), filtrado, "o-", label="Filtrado", color="royalblue")
plt.plot(range(1, len(observaciones) + 1), suavizado_lluvia, "s--", label="Suavizado", color="orange")
plt.title("Comparación: Filtrado vs Suavizado")
plt.xlabel("Día")
plt.ylabel("P(Lluvia)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# ---------------------------------------------------------
# 11. Predicción futura (+1 día)
# ---------------------------------------------------------
ultima_creencia = forward[-1]  # distribución en el último día observado
prediccion_futura = {
    "Lluvia": (
        ultima_creencia["Lluvia"]  * P_transicion["Lluvia"]["Lluvia"] +
        ultima_creencia["Soleado"] * P_transicion["Soleado"]["Lluvia"]
    ),
    "Soleado": (
        ultima_creencia["Lluvia"]  * P_transicion["Lluvia"]["Soleado"] +
        ultima_creencia["Soleado"] * P_transicion["Soleado"]["Soleado"]
    )
}
prediccion_futura = normalizar(prediccion_futura)

print("\n==============================================")
print("PREDICCIÓN FUTURA (+1 día sin evidencia nueva)")
print("==============================================")
print(f"P(Lluvia mañana) ≈ {prediccion_futura['Lluvia']:.3f}")
print(f"P(Soleado mañana) ≈ {prediccion_futura['Soleado']:.3f}")


# Comentarios:
# - El suavizado (línea naranja) ajusta las creencias pasadas
#   al considerar la evidencia futura.
# - Por ejemplo, aunque el día 3 (Seco) bajó mucho en el filtrado,
#   al ver luego más días “Mojado”, el suavizado reevalúa
#   y eleva su probabilidad de lluvia retrospectiva.
