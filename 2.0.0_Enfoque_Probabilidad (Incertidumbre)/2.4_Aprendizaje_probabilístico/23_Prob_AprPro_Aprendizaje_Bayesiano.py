# =========================================================
# 23 - APRENDIZAJE BAYESIANO
# ---------------------------------------------------------
# Descripción:
#   En el Aprendizaje Bayesiano actualizamos una creencia
#   previa P(H) sobre una hipótesis H (por ejemplo, "el clima
#   es lluvioso") en función de la evidencia observada E
#   (por ejemplo, "el suelo está mojado").
#
#   Usamos la regla de Bayes:
#
#       P(H|E) = [ P(E|H) * P(H) ] / P(E)
#
#   En este script se simula el proceso de aprendizaje
#   de un modelo que actualiza su creencia sobre si un
#   sensor está defectuoso o no, conforme llegan nuevas
#   mediciones de error o éxito.
#
#   Este enfoque es la base de los métodos de inferencia
#   bayesiana usados en machine learning probabilístico.
#
# =========================================================

from typing import List
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Definimos el modelo de hipótesis
# ---------------------------------------------------------
# Hipótesis H:
#   H0: El sensor es confiable
#   H1: El sensor está defectuoso

# Probabilidad inicial (prior)
P_H0 = 0.9   # confiable
P_H1 = 0.1   # defectuoso

# Probabilidades condicionales:
# P(lectura correcta | H)
P_correcta_dado_H0 = 0.95
P_correcta_dado_H1 = 0.6

# ---------------------------------------------------------
# 2. Evidencias observadas (mediciones)
# ---------------------------------------------------------
# Supongamos que recibimos una secuencia de observaciones:
# "C" = lectura correcta, "E" = error en la lectura
observaciones = ["C", "C", "E", "C", "E", "E", "C", "C", "E", "E"]

# ---------------------------------------------------------
# 3. Actualización Bayesiana paso a paso
# ---------------------------------------------------------
def actualizar_bayes(P_H0, P_H1, observacion):
    """Actualiza la creencia según la observación actual."""
    if observacion == "C":
        P_E_dado_H0 = P_correcta_dado_H0
        P_E_dado_H1 = P_correcta_dado_H1
    else:  # observación = "E" (error)
        P_E_dado_H0 = 1 - P_correcta_dado_H0
        P_E_dado_H1 = 1 - P_correcta_dado_H1

    # Regla de Bayes
    P_E = P_E_dado_H0 * P_H0 + P_E_dado_H1 * P_H1
    P_H0_nueva = (P_E_dado_H0 * P_H0) / P_E
    P_H1_nueva = (P_E_dado_H1 * P_H1) / P_E

    return P_H0_nueva, P_H1_nueva

# ---------------------------------------------------------
# 4. Ejecución del aprendizaje Bayesiano
# ---------------------------------------------------------
historial_H0 = [P_H0]
historial_H1 = [P_H1]

print("==============================================")
print("TRACE APRENDIZAJE BAYESIANO")
print("==============================================")

for i, obs in enumerate(observaciones, start=1):
    P_H0, P_H1 = actualizar_bayes(P_H0, P_H1, obs)
    historial_H0.append(P_H0)
    historial_H1.append(P_H1)
    print(f"Obs {i:02d}: {obs} -> P(sensor confiable)={P_H0:.4f}, P(defectuoso)={P_H1:.4f}")

# ---------------------------------------------------------
# 5. Resultados finales
# ---------------------------------------------------------
print("\n==============================================")
print("RESULTADOS FINALES")
print("==============================================")
print(f"Probabilidad final de sensor confiable : {P_H0:.4f}")
print(f"Probabilidad final de sensor defectuoso : {P_H1:.4f}")

# ---------------------------------------------------------
# 6. Gráfica de evolución de la creencia
# ---------------------------------------------------------
plt.figure(figsize=(8,4))
plt.plot(historial_H0, label="P(sensor confiable)")
plt.plot(historial_H1, label="P(sensor defectuoso)")
plt.title("Evolución de la creencia Bayesiana")
plt.xlabel("Número de observaciones")
plt.ylabel("Probabilidad")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()
