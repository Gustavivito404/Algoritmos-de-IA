# =========================================================
# 21 - RED BAYESIANA DINÁMICA: FILTRADO DE PARTÍCULAS
# ---------------------------------------------------------
# Descripción:
#   Seguimiento (tracking) de la posición de un objeto en 1D
#   usando un Filtro de Partículas (Particle Filter).
#
#   A diferencia del Filtro de Kalman, no se asume linealidad
#   ni ruido gaussiano. En su lugar, la creencia sobre el estado
#   se representa mediante un conjunto de N partículas.
#
#   Cada partícula representa una hipótesis del estado (posición)
#   y tiene un peso asociado que indica su probabilidad relativa.
#
#   Flujo general del algoritmo:
#       (1) Inicialización:
#           Crear N partículas distribuidas según una creencia inicial.
#
#       (2) Predicción:
#           Mover cada partícula según el modelo del sistema + ruido.
#
#       (3) Actualización:
#           Calcular los pesos según la probabilidad de la medición real.
#
#       (4) Re-muestreo (Resampling):
#           Seleccionar partículas proporcionalmente a sus pesos.
#           Las más probables “sobreviven”, las menos probables desaparecen.
#
#       (5) Estimación:
#           Calcular la posición estimada como el promedio ponderado
#           de todas las partículas.
#
#   Aplicaciones comunes:
#       - Robótica móvil (localización Monte Carlo)
#       - Rastreo de objetos en visión computacional
#       - Sistemas no lineales o con ruido no gaussiano
#
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. CONFIGURACIÓN DEL SISTEMA
# ---------------------------------------------------------
NUM_PARTICULAS = 500     # cantidad de partículas
NUM_PASOS = 30           # pasos de simulación
VELOCIDAD_REAL = 1.0     # m/s
POS_INICIAL_REAL = 0.0   # posición inicial verdadera
SIGMA_MOV = 0.5          # ruido del movimiento (predicción)
SIGMA_SENSOR = 1.0       # ruido del sensor (medición)

# ---------------------------------------------------------
# 2. SIMULACIÓN DEL "MUNDO REAL"
# ---------------------------------------------------------
def simular_mundo(num_pasos, vel_real, pos_inicial):
    pos = pos_inicial
    posiciones = []
    mediciones = []
    for _ in range(num_pasos):
        pos += vel_real + np.random.normal(0, 0.1)  # pequeño ruido de modelo
        medicion = pos + np.random.normal(0, SIGMA_SENSOR)
        posiciones.append(pos)
        mediciones.append(medicion)
    return np.array(posiciones), np.array(mediciones)

# ---------------------------------------------------------
# 3. FILTRADO DE PARTÍCULAS
# ---------------------------------------------------------
def inicializar_particulas(n, rango=(0, 10)):
    """Crea n partículas uniformemente distribuidas en el rango dado."""
    return np.random.uniform(rango[0], rango[1], n)

def predecir_particulas(particulas, velocidad):
    """Actualiza las partículas según el modelo de movimiento + ruido."""
    ruido_mov = np.random.normal(0, SIGMA_MOV, len(particulas))
    return particulas + velocidad + ruido_mov

def actualizar_pesos(particulas, medicion):
    """Calcula pesos según la probabilidad de la medición."""
    # Distribución gaussiana: mayor peso si la partícula coincide con la medición
    errores = medicion - particulas
    pesos = np.exp(- (errores**2) / (2 * SIGMA_SENSOR**2))
    pesos += 1e-300  # evita división por cero
    return pesos / np.sum(pesos)

def remuestrear(particulas, pesos):
    """Selecciona nuevas partículas según sus pesos (resampling)."""
    indices = np.random.choice(len(particulas), size=len(particulas), p=pesos)
    return particulas[indices]

# ---------------------------------------------------------
# 4. BUCLE PRINCIPAL
# ---------------------------------------------------------
np.random.seed(2)
pos_reales, mediciones = simular_mundo(NUM_PASOS, VELOCIDAD_REAL, POS_INICIAL_REAL)

particulas = inicializar_particulas(NUM_PARTICULAS, rango=(-5, 5))
pesos = np.ones(NUM_PARTICULAS) / NUM_PARTICULAS

estimaciones = []

for t in range(NUM_PASOS):
    # Predicción: mover partículas según el modelo
    particulas = predecir_particulas(particulas, VELOCIDAD_REAL)

    # Actualización: ajustar pesos según la medición real
    pesos = actualizar_pesos(particulas, mediciones[t])

    # Re-muestreo: mantener partículas con alto peso
    particulas = remuestrear(particulas, pesos)

    # Estimación: promedio ponderado
    estimacion = np.mean(particulas)
    estimaciones.append(estimacion)

    # Mostrar traza cada cierto paso
    if t % 5 == 0 or t == NUM_PASOS - 1:
        print(f"[Paso {t:02d}] Medición={mediciones[t]:.2f} | "
              f"Estimado={estimacion:.2f} | Real={pos_reales[t]:.2f}")

# ---------------------------------------------------------
# 5. RESULTADOS Y GRÁFICAS
# ---------------------------------------------------------
plt.figure(figsize=(10,5))
plt.plot(pos_reales, label="Posición real", linewidth=2)
plt.plot(mediciones, 'o', label="Mediciones (sensor)", alpha=0.5)
plt.plot(estimaciones, '--', label="Estimación (Filtro de Partículas)", linewidth=2)
plt.title("Filtro de Partículas - Rastreo de posición 1D")
plt.xlabel("Paso de tiempo")
plt.ylabel("Posición")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

print("\n==============================================")
print("RESULTADOS FINALES")
print("==============================================")
print(f"Posición real final: {pos_reales[-1]:.2f}")
print(f"Estimación final    : {estimaciones[-1]:.2f}")