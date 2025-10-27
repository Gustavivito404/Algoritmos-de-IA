# =========================================================
# 12 - MUESTREO DIRECTO Y POR RECHAZO
# ---------------------------------------------------------
# Descripción:
#   En este script implementamos dos métodos de inferencia
#   aproximada en Redes Bayesianas:
#
#     1. Muestreo Directo (Direct Sampling)
#     2. Muestreo por Rechazo (Rejection Sampling)
#
#   En lugar de calcular todas las combinaciones posibles
#   (como en Enumeración o Eliminación de Variables),
#   aquí generamos muestras aleatorias del "mundo"
#   siguiendo la estructura causal de la red.
#
#   Luego estimamos probabilidades observando
#   cuántas muestras coinciden con nuestra evidencia.
#
# Características:
#   - Mucho más rápido en redes grandes.
#   - Pero introduce error estadístico (depende del número de muestras).
#   - Si la evidencia es muy poco probable, el muestreo por rechazo
#     puede ser muy ineficiente.
#
# Ejemplo visual (misma red):
#
#     Lluvia  ──►  SensorAgua  ──►  Alarma
#
#   Queremos estimar:
#       P(Lluvia | Alarma=True)
#
# =========================================================

import random
from typing import Dict, Tuple
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Probabilidades base de la red
# ---------------------------------------------------------
P_Lluvia = {True: 0.3, False: 0.7}
P_SensorAgua_dado_Lluvia = {
    (True,  True): 0.9,
    (False, True): 0.1,
    (True,  False): 0.2,
    (False, False): 0.8
}
P_Alarma_dado_Sensor = {
    (True,  True): 0.95,
    (False, True): 0.05,
    (True,  False): 0.01,
    (False, False): 0.99
}

# ---------------------------------------------------------
# 2. Funciones auxiliares para muestreo
# ---------------------------------------------------------
def sample_boolean(prob_true: float) -> bool:
    """Devuelve True con probabilidad 'prob_true'."""
    return random.random() < prob_true

def generar_muestra_directa() -> Dict[str, bool]:
    """
    Genera UNA muestra del mundo (Lluvia, SensorAgua, Alarma)
    siguiendo el orden causal de la red bayesiana.
    """
    # 1) Muestreamos Lluvia directamente de su probabilidad
    lluvia = sample_boolean(P_Lluvia[True])

    # 2) Muestreamos SensorAgua condicionado en Lluvia
    p_sensor = P_SensorAgua_dado_Lluvia[(True, lluvia)]
    sensor = sample_boolean(p_sensor)

    # 3) Muestreamos Alarma condicionado en SensorAgua
    p_alarma = P_Alarma_dado_Sensor[(True, sensor)]
    alarma = sample_boolean(p_alarma)

    return {"Lluvia": lluvia, "SensorAgua": sensor, "Alarma": alarma}

# ---------------------------------------------------------
# 3. Muestreo Directo
# ---------------------------------------------------------
def muestreo_directo(n_muestras: int, evidencia: Dict[str,bool], query: str) -> float:
    """
    Genera muestras directamente de la red y estima P(query | evidencia).
    """
    muestras_validas = []
    for _ in range(n_muestras):
        muestra = generar_muestra_directa()
        # Verificamos si cumple la evidencia
        if all(muestra[var] == val for var, val in evidencia.items()):
            muestras_validas.append(muestra)

    if not muestras_validas:
        return 0.0

    # Estimamos P(query=True | evidencia)
    cuenta_true = sum(1 for m in muestras_validas if m[query] is True)
    return cuenta_true / len(muestras_validas)

# ---------------------------------------------------------
# 4. Muestreo por Rechazo
# ---------------------------------------------------------
def muestreo_por_rechazo(n_muestras: int, evidencia: Dict[str,bool], query: str) -> float:
    """
    Similar al directo, pero aquí generamos muestras completas
    y rechazamos las que NO cumplen la evidencia.
    """
    cuenta_total = 0
    cuenta_true = 0

    for _ in range(n_muestras):
        muestra = generar_muestra_directa()

        # Rechazamos si no cumple la evidencia
        if not all(muestra[var] == val for var, val in evidencia.items()):
            continue

        cuenta_total += 1
        if muestra[query] is True:
            cuenta_true += 1

    if cuenta_total == 0:
        return 0.0
    return cuenta_true / cuenta_total

# ---------------------------------------------------------
# 5. Ejecución y comparación
# ---------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)

    print("==============================================")
    print("TRACE MUESTREO DIRECTO Y POR RECHAZO")
    print("==============================================\n")

    N = 10000
    evidencia = {"Alarma": True}
    query = "Lluvia"

    print(f"Generando {N} muestras del mundo...\n")

    p_directo = muestreo_directo(N, evidencia, query)
    p_rechazo = muestreo_por_rechazo(N, evidencia, query)

    print("Resultados:")
    print(f"   P(Lluvia=True | Alarma=True) [Muestreo Directo]  ≈ {p_directo:.4f}")
    print(f"   P(Lluvia=True | Alarma=True) [Muestreo Rechazo] ≈ {p_rechazo:.4f}\n")

    print("Interpretación:")
    print("- Ambos métodos usan simulación Monte Carlo para estimar la probabilidad.")
    print("- El muestreo directo usa la estructura de la red para generar mundos coherentes.")
    print("- El muestreo por rechazo genera y descarta muestras que no cumplen la evidencia.")
    print("- Con suficientes muestras, ambos convergen al valor exacto (~0.66 en este caso).")

    # ---------------------------------------------------------
    # 6. (OPCIONAL) Gráfica comparativa
    # ---------------------------------------------------------
    # Solo se genera si lo consideras útil.
    #
    # plt.bar(["Directo","Rechazo"], [p_directo, p_rechazo], color=["skyblue","salmon"])
    # plt.ylim(0,1)
    # plt.title("Estimación de P(Lluvia | Alarma=True)")
    # plt.ylabel("Probabilidad estimada")
    # plt.show()
    #
    # (Descomenta para visualizar la comparación)
