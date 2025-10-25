# =========================================================
# 31 - RED BAYESIANA DINÁMICA (Dynamic Bayesian Network)
# ---------------------------------------------------------
# Descripción:
#   Simulación simple de una DBN con dos variables:
#   - X_t: estado oculto (posición del robot)
#   - O_t: observación ruidosa (sensor)
#
#   Se modela:
#     P(X_t | X_{t-1}) : transición
#     P(O_t | X_t)     : observación
#
#   Y se realiza filtrado temporal:
#     P(X_t | O_1:t)   : creencia actualizada con observaciones
#
# =========================================================

import numpy as np

# ---------------------------------------------------------
# 1. Definición de los posibles estados y observaciones
# ---------------------------------------------------------
# Supongamos que el robot puede estar en tres posiciones:
#  X ∈ {izquierda, centro, derecha}
# Y los sensores solo reportan "izquierda" o "derecha" con ruido.

estados = ["izquierda", "centro", "derecha"]
observaciones = ["izquierda", "derecha"]

# ---------------------------------------------------------
# 2. Modelo de transición P(X_t | X_{t-1})
# ---------------------------------------------------------
# El robot tiende a quedarse donde está, pero puede moverse con probabilidad.
P_transicion = np.array([
    [0.7, 0.3, 0.0],  # desde izquierda
    [0.2, 0.6, 0.2],  # desde centro
    [0.0, 0.3, 0.7]   # desde derecha
])

# ---------------------------------------------------------
# 3. Modelo de observación P(O_t | X_t)
# ---------------------------------------------------------
# El sensor tiene ruido: a veces confunde izquierda con derecha.
P_observacion = np.array([
    [0.9, 0.1],  # si está en izquierda
    [0.5, 0.5],  # si está en centro
    [0.1, 0.9]   # si está en derecha
])

# ---------------------------------------------------------
# 4. Creencia inicial
# ---------------------------------------------------------
b = np.array([0.3, 0.4, 0.3])  # distribución inicial sobre X₀

def normalizar(p):
    return p / np.sum(p)

# ---------------------------------------------------------
# 5. Función de filtrado Bayesiano (Forward)
# ---------------------------------------------------------
def actualizar_creencia(b_anterior, observacion_idx):
    """
    Calcula b_t = α * P(O_t | X_t) * Σ P(X_t|X_{t-1}) * b_{t-1}
    """
    # Predicción: P(X_t) = Σ_s P(X_t|X_{t-1}) * b_{t-1}(s)
    pred = P_transicion.T @ b_anterior

    # Actualización con observación
    like = P_observacion[:, observacion_idx]
    b_nueva = like * pred

    # Normalización
    b_nueva = normalizar(b_nueva)
    return b_nueva

# ---------------------------------------------------------
# 6. Simulación paso a paso
# ---------------------------------------------------------
if __name__ == "__main__":
    print("==============================================")
    print("TRACE RED BAYESIANA DINÁMICA (DBN)")
    print("==============================================\n")

    # Secuencia de observaciones simuladas
    obs_seq = ["izquierda", "derecha", "derecha", "izquierda"]
    print(f"Secuencia de observaciones: {obs_seq}\n")

    b_actual = b
    print(f"Creencia inicial: {dict(zip(estados, b_actual.round(3)))}\n")

    for t, obs in enumerate(obs_seq, 1):
        idx_obs = observaciones.index(obs)
        b_actual = actualizar_creencia(b_actual, idx_obs)
        print(f"[t={t}] Observación: {obs}")
        for i, estado in enumerate(estados):
            print(f"   P({estado}|O₁:{t}) = {b_actual[i]:.3f}")
        print("")

    # Nota:
    # - Esta simulación muestra cómo una DBN propaga creencias en el tiempo.
    # - La creencia "se mueve" hacia donde las observaciones sugieren,
    #   pero suavizada por el modelo de transición (no cambia bruscamente).
