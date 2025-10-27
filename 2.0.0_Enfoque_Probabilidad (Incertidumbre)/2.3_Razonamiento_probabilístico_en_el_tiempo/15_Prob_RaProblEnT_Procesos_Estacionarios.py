# =========================================================
# 15 - PROCESOS ESTACIONARIOS
# ---------------------------------------------------------
# Descripción:
#   Un proceso estacionario es aquel en el que las
#   probabilidades de transición entre estados no cambian
#   con el tiempo.
#
#   Es decir:
#       P(X_t+1 | X_t) = constante
#
#   En nuestro ejemplo, modelamos el clima (lluvia o no lluvia)
#   como un proceso de Markov estacionario:
#
#       X_t ∈ {Lluvia, Soleado}
#
#   La probabilidad de que mañana llueva depende solo de
#   si hoy llueve, y las reglas de transición no cambian.
#
#   Este es el punto de partida para todos los modelos
#   probabilísticos en el tiempo.
#
# Grafo del proceso:
#
#       Día t       Día t+1
#       ------      -------
#        Lluvia ─────► Lluvia
#          │             │
#          ▼             ▼
#       Soleado ─────► Soleado
#
#   Cada flecha tiene una probabilidad fija.
# =========================================================

import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Definimos los estados y las probabilidades de transición
# ---------------------------------------------------------
# P(X_{t+1} | X_t)
P_transicion = {
    "Lluvia":   {"Lluvia": 0.7, "Soleado": 0.3},
    "Soleado":  {"Lluvia": 0.4, "Soleado": 0.6}
}

# Estado inicial (probabilidad de lluvia al inicio)
P_inicial = {"Lluvia": 0.5, "Soleado": 0.5}

# ---------------------------------------------------------
# 2. Evolución del proceso en el tiempo
# ---------------------------------------------------------
def evolucionar_proceso(P_actual):
    """Calcula la distribución del siguiente día."""
    P_siguiente = {}
    for estado_siguiente in ["Lluvia", "Soleado"]:
        P_siguiente[estado_siguiente] = (
            P_actual["Lluvia"]  * P_transicion["Lluvia"][estado_siguiente] +
            P_actual["Soleado"] * P_transicion["Soleado"][estado_siguiente]
        )
    return P_siguiente

# ---------------------------------------------------------
# 3. Simulación a lo largo del tiempo
# ---------------------------------------------------------
num_dias = 100
historial_lluvia = [P_inicial["Lluvia"]]
P_actual = P_inicial

print("==============================================")
print("TRACE PROCESO ESTACIONARIO (Evolución del clima)")
print("==============================================\n")
print(f"Día 0: P(Lluvia) = {P_actual['Lluvia']:.3f}")

for t in range(1, num_dias+1):
    P_actual = evolucionar_proceso(P_actual)
    historial_lluvia.append(P_actual["Lluvia"])
    print(f"Día {t}: P(Lluvia) = {P_actual['Lluvia']:.3f}")

# ---------------------------------------------------------
# 4. Resultado final e interpretación
# ---------------------------------------------------------
print("\n==============================================")
print("RESULTADO FINAL")
print("==============================================")
print(f"Distribución final después de {num_dias} días:")
print(f"P(Lluvia) = {P_actual['Lluvia']:.4f}")
print(f"P(Soleado) = {P_actual['Soleado']:.4f}")
print("\nInterpretación:")
print("- El proceso llega a un punto de equilibrio llamado")
print("  'distribución estacionaria'.")
print("- Después de muchos días, la probabilidad de lluvia")
print(f"  deja de cambiar (≈ {P_actual['Lluvia']:.3f} en este ejemplo).")
print("- Esto significa que el sistema se estabilizó de acuerdo")
print("  a las probabilidades de transición que definimos.")
print("")
# Nota:
#   El valor estacionario se puede calcular analíticamente como:
#   p* = b / (1 - a + b)
#   donde:
#       a = P(Lluvia mañana | Lluvia hoy)
#       b = P(Lluvia mañana | Soleado hoy)
#   En este código:
#       a = 0.7
#       b = 0.4
#       p* ≈ 0.5714


# ---------------------------------------------------------
# 5. (OPCIONAL) Gráfica de convergencia
# ---------------------------------------------------------
# La gráfica muestra cómo la probabilidad de lluvia
# se estabiliza con el tiempo.
plt.plot(range(num_dias+1), historial_lluvia, marker="o", color="royalblue")
plt.title("Evolución de P(Lluvia) en un proceso estacionario")
plt.xlabel("Día")
plt.ylabel("Probabilidad de lluvia")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
