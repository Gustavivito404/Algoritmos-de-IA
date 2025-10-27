# =========================================================
# 20 - FILTRO DE KALMAN (1D posición + velocidad)
# ---------------------------------------------------------
# Descripción:
#   Estimamos posición y velocidad reales de un objeto
#   en 1D usando un Filtro de Kalman lineal discreto.
#
#   Estado oculto:
#       x = [posicion,
#            velocidad]^T   (vector 2x1)
#
#   Dinámica (modelo del mundo):
#       pos_{t+1} = pos_t + vel_t * dt
#       vel_{t+1} = vel_t
#
#       x_{t+1} = A x_t + w_t
#       w_t ~ N(0, Q)
#
#   Sensor:
#       z_t = H x_t + v_t
#       v_t ~ N(0, R)
#
#   Donde H "mide" solo la posición.
#
#   Flujo Kalman por paso k:
#       (1) Predicción:
#           x_pred = A x_est
#           P_pred = A P_est A^T + Q
#
#       (2) Actualización c/medición z_k:
#           y     = z_k - H x_pred                  (innovación)
#           S     = H P_pred H^T + R                (varianza innovación)
#           K     = P_pred H^T S^{-1}               (ganancia de Kalman)
#           x_est = x_pred + K y                    (nuevo estado)
#           P_est = (I - K H) P_pred                (nueva covarianza)
#
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# ---------------------------------------------------------
# 1. Parámetros del modelo dinámico
# ---------------------------------------------------------
dt = 1.0  # paso de muestreo (1s)

# A: transición de estado (2x2)
A = np.array([
    [1, dt],
    [0, 1]
])

# H: matriz de observación (1x2)
# z = H x -> medimos solo la posición
H = np.array([
    [1, 0]
])

# Q: covarianza del ruido del proceso (2x2)
Q = np.array([
    [0.01, 0.0],
    [0.0,  0.01]
])

# R: covarianza del ruido de medición (1x1)
R = np.array([[0.5]])

# Identidad 2x2 (la usamos en update)
I = np.eye(2)


# ---------------------------------------------------------
# 2. Simulación del "mundo real" + mediciones ruidosas
# ---------------------------------------------------------
def simular_mundo(num_pasos: int,
                  x0_real: np.ndarray,
                  vel_real: float,
                  ruido_medicion_std: float = 1.0) -> Tuple[List[float], List[float]]:
    """
    Simula el movimiento real (sin ruido dinámico para que sea didáctico)
    y genera mediciones ruidosas de posición.

    x0_real: np.array([pos0, vel0])  <-- usado sólo para inicializar
    vel_real: velocidad constante "verdadera"
    ruido_medicion_std: sigma del ruido gaussiano del sensor
    """
    posiciones_reales = []
    mediciones = []

    pos = x0_real[0]
    vel = vel_real

    for k in range(num_pasos):
        # guardar la "verdad"
        posiciones_reales.append(pos)

        # medición ruidosa de posición
        z = pos + np.random.normal(0, ruido_medicion_std)
        mediciones.append(z)

        # actualizamos el mundo real
        pos = pos + vel * dt  # mvto uniforme

    return posiciones_reales, mediciones


# ---------------------------------------------------------
# 3. Paso de predicción del filtro de Kalman
# ---------------------------------------------------------
def kalman_predict(x_est: np.ndarray,
                   P_est: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predicción de siguiente estado:
    x_est: (2x1)
    P_est: (2x2)
    Devuelve x_pred (2x1), P_pred (2x2)
    """
    # x^- = A x^+
    x_pred = A @ x_est  # (2x2)(2x1) -> (2x1)

    # P^- = A P^+ A^T + Q
    P_pred = A @ P_est @ A.T + Q  # (2x2)(2x2)(2x2) -> (2x2)

    return x_pred, P_pred


# ---------------------------------------------------------
# 4. Paso de actualización con medición
# ---------------------------------------------------------
def kalman_update(x_pred: np.ndarray,
                  P_pred: np.ndarray,
                  z_medida: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Actualización usando la observación z_medida (escalar).
    Devuelve x_est (2x1), P_est (2x2)
    """
    # Innovación: y = z - H x^-
    # H (1x2), x_pred (2x1) -> (1x1)
    y = z_medida - (H @ x_pred)[0, 0]

    # S = H P^- H^T + R   -> (1x1)
    S = H @ P_pred @ H.T + R

    # K = P^- H^T S^{-1}  -> (2x2)(2x1)(1x1)^-1 -> (2x1)
    K = P_pred @ H.T @ np.linalg.inv(S)

    # x^+ = x^- + K y
    x_est = x_pred + K * y  # (2x1) + (2x1)*escalar -> (2x1)

    # P^+ = (I - K H) P^-
    P_est = (I - K @ H) @ P_pred  # (2x2)(2x2)->(2x2)

    return x_est, P_est


# ---------------------------------------------------------
# 5. MAIN con traza paso a paso
# ---------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(1)

    # Configuración de la simulación
    NUM_PASOS = 20
    VELOCIDAD_REAL = 1.0      # m/s constante real
    POS_INICIAL_REAL = 0.0    # arranca en x = 0.0

    # Estado real inicial (para simular mundo)
    x0_real = np.array([POS_INICIAL_REAL, VELOCIDAD_REAL])

    # Simular mundo verdadero y sensor ruidoso
    posiciones_reales, mediciones = simular_mundo(
        num_pasos=NUM_PASOS,
        x0_real=x0_real,
        vel_real=VELOCIDAD_REAL,
        ruido_medicion_std=1.0
    )

    # Estimación inicial del Filtro de Kalman:
    # La guardamos en forma de vector columna (2x1)
    # para que toda la álgebra sea consistente.
    x_est = np.array([[5.0],
                      [0.0]])   # empezamos creyendo pos=5, vel=0 (mal a propósito)

    # Covarianza inicial (2x2) grande -> mucha incertidumbre
    P_est = np.array([
        [10.0, 0.0],
        [0.0, 10.0]
    ])

    print("==============================================")
    print("TRACE FILTRO DE KALMAN")
    print("==============================================\n")

    print(f"Estado inicial REAL:     pos={x0_real[0]:.2f}, vel={x0_real[1]:.2f}")
    print(f"Estado inicial ESTIMADO: pos={x_est[0,0]:.2f}, vel={x_est[1,0]:.2f}")
    print(f"Covarianza inicial P:\n{P_est}\n")

    historial_pos_real = []
    historial_pos_medida = []
    historial_pos_estimada = []
    historial_vel_estimada = []

    # Iteramos en el tiempo
    for k in range(NUM_PASOS):
        z = mediciones[k]          # medición ruidosa del sensor (posición)
        real_pos = posiciones_reales[k]

        historial_pos_real.append(real_pos)
        historial_pos_medida.append(z)

        # -------- PREDICCIÓN --------
        x_pred, P_pred = kalman_predict(x_est, P_est)

        # -------- ACTUALIZACIÓN -----
        x_est, P_est = kalman_update(x_pred, P_pred, z)

        # Guardar estimaciones para gráfica
        historial_pos_estimada.append(x_est[0,0])
        historial_vel_estimada.append(x_est[1,0])

        # Traza legible
        print(f"[Paso {k:02d}]")
        print(f"  Medición z = {z:.2f}  (posición ruidosa)")
        print(f"  Real      : pos={real_pos:.2f}")
        print(f"  Predicción: pos={x_pred[0,0]:.2f}, vel={x_pred[1,0]:.2f}")
        print(f"  Corrección: pos={x_est[0,0]:.2f}, vel={x_est[1,0]:.2f}")
        print(f"  P_est (covarianza actualizada):\n{P_est}\n")

    print("==============================================")
    print("RESUMEN FINAL")
    print("==============================================")
    print(f"Última posición real     : {historial_pos_real[-1]:.2f}")
    print(f"Última medición ruidosa  : {historial_pos_medida[-1]:.2f}")
    print(f"Última posición estimada : {historial_pos_estimada[-1]:.2f}")
    print(f"Última velocidad estimada: {historial_vel_estimada[-1]:.2f}")

    # -------------------------------------------------
    # 6. Gráficas opcionales
    # -------------------------------------------------
    pasos = list(range(NUM_PASOS))

    plt.figure(figsize=(9,4))
    plt.plot(pasos, historial_pos_real,     label="Posición real", linewidth=2)
    plt.plot(pasos, historial_pos_medida,   label="Medición (ruido)", linestyle="dotted")
    plt.plot(pasos, historial_pos_estimada, label="Estimación Kalman", linestyle="dashdot")
    plt.title("Filtro de Kalman - Rastreo de posición")
    plt.xlabel("Paso de tiempo")
    plt.ylabel("Posición")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()

    plt.figure(figsize=(9,4))
    plt.plot(pasos, historial_vel_estimada, label="Velocidad estimada (Kalman)", linewidth=2)
    plt.axhline(VELOCIDAD_REAL, color="gray", linestyle="dashed", label="Velocidad real")
    plt.title("Filtro de Kalman - Estimación de velocidad")
    plt.xlabel("Paso de tiempo")
    plt.ylabel("Velocidad")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()

    # Nota:
    #   - x_est ahora es SIEMPRE un vector columna 2x1.
    #   - Eso evita problemas de formas raras (flatten).
    #   - La multiplicación matricial A @ x_est ya cuadra.
    #   - Y al imprimir usamos x_est[0,0] / x_est[1,0] que son escalares.
