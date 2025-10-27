# =========================================================
# 16 - HIPÓTESIS DE MARKOV: PROCESOS DE MARKOV
# ---------------------------------------------------------
# Descripción:
#   Este script implementa un proceso de Markov simple
#   (cadena de dos estados: Lluvia y Soleado).
#
#   La Hipótesis de Markov establece que:
#       P(X_t+1 | X_t, X_t-1, ..., X_0) = P(X_t+1 | X_t)
#
#   Es decir, el siguiente estado depende únicamente
#   del estado actual, no de toda la historia.
#
#   Simulamos una secuencia de días con transición
#   probabilística entre los estados, y medimos
#   la frecuencia empírica para verificar que se acerca
#   al equilibrio teórico (~0.571 lluvia).
#
# Grafo conceptual:
#
#        ┌───────┐ 0.7 ┌───────┐
#        │ Lluvia│────►│ Lluvia│
#        └───────┘     └───────┘
#           │0.3          ▲0.4
#           ▼             │
#        ┌───────┐ 0.6 ┌───────┐
#        │Soleado│────►│Soleado│
#        └───────┘     └───────┘
#
# =========================================================

import random
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Matriz de transición
# ---------------------------------------------------------
P_transicion = {
    "Lluvia":   {"Lluvia": 0.7, "Soleado": 0.3},
    "Soleado":  {"Lluvia": 0.4, "Soleado": 0.6}
}

# ---------------------------------------------------------
# 2. Parámetros de simulación
# ---------------------------------------------------------
num_dias = 200
estado_actual = "Lluvia"  # podemos empezar con cualquier estado
historial = [estado_actual]

print("==============================================")
print("TRACE PROCESO DE MARKOV (Simulación de estados)")
print("==============================================")

for t in range(1, num_dias + 1):
    p_lluvia = P_transicion[estado_actual]["Lluvia"]
    # Generamos un número aleatorio para decidir transición
    if random.random() < p_lluvia:
        estado_actual = "Lluvia"
    else:
        estado_actual = "Soleado"

    historial.append(estado_actual)
    if t <= 15:  # solo mostramos los primeros 15 días para claridad
        print(f"Día {t:2d}: {estado_actual}")

# ---------------------------------------------------------
# 3. Cálculo de frecuencias empíricas
# ---------------------------------------------------------
freq_lluvia = historial.count("Lluvia") / len(historial)
freq_soleado = historial.count("Soleado") / len(historial)

print("\n==============================================")
print("RESULTADO ESTADÍSTICO")
print("==============================================")
print(f"Días totales simulados: {len(historial)}")
print(f"Frecuencia Lluvia  ≈ {freq_lluvia:.4f}")
print(f"Frecuencia Soleado ≈ {freq_soleado:.4f}")
print("\nComparación con el equilibrio teórico:")
print("p* = b / (1 - a + b) = 0.4 / (1 - 0.7 + 0.4) = 0.5714")

# ---------------------------------------------------------
# 4. Visualización (opcional)
# ---------------------------------------------------------
# Convertimos lluvia=1, soleado=0 para graficar una línea temporal
valores = [1 if s == "Lluvia" else 0 for s in historial]

plt.figure(figsize=(8, 3))
plt.plot(valores, drawstyle="steps-pre", color="royalblue")
plt.title("Simulación de un Proceso de Markov (Clima)")
plt.xlabel("Día")
plt.ylabel("Lluvia = 1 | Soleado = 0")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Comentario final:
# - Aunque la secuencia es aleatoria (ruidosa),
#   la frecuencia promedio de lluvia converge hacia ~0.571.
# - Esto ilustra perfectamente la Hipótesis de Markov:
#   el futuro depende solo del presente, no del pasado completo.
