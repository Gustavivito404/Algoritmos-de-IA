# =========================================================
# 13 - PONDERACIÓN DE VEROSIMILITUD
# ---------------------------------------------------------
# Descripción:
#   En este script implementamos "Ponderación de Verosimilitud"
#   (Likelihood Weighting), otra técnica de inferencia aproximada
#   en Redes Bayesianas.
#
#   Objetivo:
#     Estimar P(Query | Evidencia)
#     incluso cuando la evidencia tiene baja probabilidad.
#
#   Problema que resuelve:
#     - En muestreo por rechazo (script 12), tirábamos muestras
#       que no coincidían con la evidencia.
#     - Si la evidencia es rara, rechazamos casi TODO y nos quedamos
#       con muy pocas muestras útiles → mala estimación.
#
#   Idea de la Ponderación de Verosimilitud:
#     1) Forzamos que la evidencia siempre se cumpla.
#        (no rechazamos esas muestras)
#     2) PERO cada muestra recibe un peso w,
#        que refleja qué tan "probable" era esa evidencia
#        bajo la configuración generada.
#
#   Ejemplo visual de la red (misma de antes):
#
#       Lluvia ──► SensorAgua ──► Alarma
#
#   Vamos a estimar:
#       P(Lluvia | Alarma=True)
#
# Características:
#   - Siempre produce muestras válidas con la evidencia fija.
#   - Usa pesos para corregir el sesgo.
#   - Mucho más eficiente que Rechazo cuando la evidencia
#     es poco común.
#
# Fórmula intuitiva:
#
#   Aproximamos:
#       P(Query = True | Evidencia)
#     ≈ (suma de pesos de muestras donde Query=True)
#        / (suma de pesos de TODAS las muestras)
#
#   Donde cada peso w es el producto de las probabilidades
#   de las variables de evidencia dadas sus padres.
#
# =========================================================

import random
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

# ---------------------------------------------------------
# 1. Red Bayesiana (misma estructura que en scripts 07-12)
# ---------------------------------------------------------

# P(Lluvia)
P_Lluvia = {True: 0.3, False: 0.7}

# P(SensorAgua | Lluvia)
# clave: (SensorAgua, Lluvia)
P_SensorAgua_dado_Lluvia = {
    (True,  True): 0.9,
    (False, True): 0.1,
    (True,  False): 0.2,
    (False, False): 0.8
}

# P(Alarma | SensorAgua)
# clave: (Alarma, SensorAgua)
P_Alarma_dado_Sensor = {
    (True,  True): 0.95,
    (False, True): 0.05,
    (True,  False): 0.01,
    (False, False): 0.99
}

# ---------------------------------------------------------
# 2. Funciones auxiliares
# ---------------------------------------------------------

def sample_boolean(prob_true: float) -> bool:
    """Devuelve True con probabilidad prob_true."""
    return random.random() < prob_true

def ponderacion_de_verosimilitud(N_muestras: int,
                                 evidencia: Dict[str,bool],
                                 query: str) -> Tuple[float, List[float]]:
    """
    Ponderación de Verosimilitud:
    - Generamos N_muestras.
    - Fijamos las variables de evidencia a los valores dados.
    - Para las variables SIN evidencia, las sampleamos normalmente.
    - Cada muestra obtiene un PESO w = producto de las probabilidades
      de la evidencia condicionada a sus padres.
    - Estimamos P(query=True | evidencia) usando suma ponderada.

    Además: devolvemos la lista de pesos usados
    (nos va a servir para graficar distribución de pesos).
    """

    pesos = []               # para análisis gráfico opcional
    peso_total_true = 0.0    # suma de pesos de muestras donde query=True
    peso_total_all = 0.0     # suma de pesos de TODAS las muestras

    for _ in range(N_muestras):
        # Vamos a muestrear en ORDEN TOPOLOGICO:
        # Lluvia -> SensorAgua -> Alarma
        # Peeero:
        #  - Si variable está en evidencia, no la sorteamos, la fijamos.
        #  - Si variable está en evidencia, también multiplicamos al peso
        #    por la probabilidad de ese valor dado sus padres.

        w = 1.0   # peso acumulado de esta muestra
        asignacion = {}

        # 1) Lluvia
        if "Lluvia" in evidencia:
            # valor fijo
            lluvia_val = evidencia["Lluvia"]
            asignacion["Lluvia"] = lluvia_val

            # peso *= P(Lluvia = ese valor)
            w *= P_Lluvia[lluvia_val]
        else:
            # muestreamos normalmente
            lluvia_val = sample_boolean(P_Lluvia[True])
            asignacion["Lluvia"] = lluvia_val

        # 2) SensorAgua | Lluvia
        p_sensor_true = P_SensorAgua_dado_Lluvia[(True, lluvia_val)]
        if "SensorAgua" in evidencia:
            sensor_val = evidencia["SensorAgua"]
            asignacion["SensorAgua"] = sensor_val

            # peso por la probabilidad de que SensorAgua tomara ese valor
            # dado Lluvia
            if sensor_val is True:
                w *= p_sensor_true
            else:
                w *= (1 - p_sensor_true)
        else:
            # sample normal
            sensor_val = sample_boolean(p_sensor_true)
            asignacion["SensorAgua"] = sensor_val

        # 3) Alarma | SensorAgua
        p_alarma_true = P_Alarma_dado_Sensor[(True, sensor_val)]
        if "Alarma" in evidencia:
            alarma_val = evidencia["Alarma"]
            asignacion["Alarma"] = alarma_val

            # actualizamos peso con la prob condicional correcta
            if alarma_val is True:
                w *= p_alarma_true
            else:
                w *= (1 - p_alarma_true)
        else:
            # sample normal
            alarma_val = sample_boolean(p_alarma_true)
            asignacion["Alarma"] = alarma_val

        # Ya tenemos:
        #  - asignacion completa para (Lluvia, SensorAgua, Alarma)
        #  - peso w que representa lo "probable" que era
        #    que la evidencia tomara esos valores

        # Actualizamos contadores ponderados:
        if asignacion[query] is True:
            peso_total_true += w
        peso_total_all += w

        pesos.append(w)

    if peso_total_all == 0:
        prob_estimada = 0.0
    else:
        prob_estimada = peso_total_true / peso_total_all

    return prob_estimada, pesos

# ---------------------------------------------------------
# 3. Comparación con muestreo por rechazo
# ---------------------------------------------------------
def muestreo_por_rechazo(N_muestras: int,
                         evidencia: Dict[str,bool],
                         query: str) -> float:
    """
    Reutilizamos la lógica básica del script anterior (12),
    pero la vuelvo a declarar aquí para que este archivo sea autónomo.
    """
    def generar_muestra_directa():
        lluvia = sample_boolean(P_Lluvia[True])
        p_sensor = P_SensorAgua_dado_Lluvia[(True, lluvia)]
        sensor = sample_boolean(p_sensor)
        p_alarma = P_Alarma_dado_Sensor[(True, sensor)]
        alarma = sample_boolean(p_alarma)
        return {"Lluvia": lluvia, "SensorAgua": sensor, "Alarma": alarma}

    cuenta_total = 0
    cuenta_true = 0
    for _ in range(N_muestras):
        m = generar_muestra_directa()
        # "rechazamos" si no cumple evidencia
        if not all(m[var] == val for var,val in evidencia.items()):
            continue
        cuenta_total += 1
        if m[query] is True:
            cuenta_true += 1

    if cuenta_total == 0:
        return 0.0
    return cuenta_true / cuenta_total

# ---------------------------------------------------------
# 4. MAIN: ejecutar prueba y traza
# ---------------------------------------------------------
if __name__ == "__main__":
    random.seed(123)

    print("==============================================")
    print("TRACE PONDERACIÓN DE VEROSIMILITUD")
    print("==============================================\n")

    N = 10000
    evidencia = {"Alarma": True}
    query = "Lluvia"

    print(f"1) Objetivo: estimar P({query}=True | {evidencia})")
    print("   Interpretación: 'Sonó la alarma. ¿Qué tan probable es que llueva?'")
    print("")

    # Muestreo por rechazo
    p_rechazo = muestreo_por_rechazo(N, evidencia, query)
    print("2) Muestreo por Rechazo:")
    print(f"   Estimación ≈ {p_rechazo:.4f}")
    print("   Nota: Este método DESCARTA muestras que no cumplen la evidencia.")
    print("         Si la evidencia es rara, casi no quedan muestras útiles.\n")

    # Ponderación de verosimilitud
    p_ponderada, pesos = ponderacion_de_verosimilitud(N, evidencia, query)
    print("3) Ponderación de Verosimilitud:")
    print(f"   Estimación ≈ {p_ponderada:.4f}")
    print("   Nota: Este método NUNCA descarta muestras.")
    print("         Siempre forzamos la evidencia (Alarma=True),")
    print("         pero le damos a cada muestra un PESO distinto según")
    print("         qué tan probable era esa evidencia.")
    print("")

    print("4) Comparación:")
    print(f"   Rechazo      -> {p_rechazo:.4f}")
    print(f"   Verosimilitud-> {p_ponderada:.4f}")
    print("   (Ambos deberían acercarse al valor 'real' ~0.66 con suficientes muestras.)")
    print("")

    # ---------------------------------------------------------
    # 5. (OPCIONAL) Visualización con matplotlib
    # ---------------------------------------------------------
    # Aquí la gráfica SÍ es útil porque:
    #   - Podemos ver la distribución de pesos 'w'.
    #   - Eso te enseña por qué la ponderación de verosimilitud
    #     puede ser numéricamente inestable si muy pocos pesos dominan.
    #
    # Vamos a generar dos gráficos:
    #   A) Histograma de pesos (cómo se reparten los w).
    #   B) Barras comparando la estimación de ambos métodos.
    #
    # Nota:
    #   Estas gráficas ayudan mucho para reportes,
    #   para mostrar "por qué" el método funciona.
    #
    # Puedes descomentar plt.show() cuando lo corras localmente.

    fig, axs = plt.subplots(1, 2, figsize=(10,4))

    # A) Histograma de pesos
    axs[0].hist(pesos, bins=30)
    axs[0].set_title("Distribución de pesos (w)\nPonderación de Verosimilitud")
    axs[0].set_xlabel("peso w")
    axs[0].set_ylabel("frecuencia")

    # B) Comparación de resultados
    axs[1].bar(["Rechazo", "Verosimilitud"], [p_rechazo, p_ponderada])
    axs[1].set_ylim(0,1)
    axs[1].set_ylabel("Prob. estimada de Lluvia")
    axs[1].set_title("Estimación de P(Lluvia | Alarma=True)")

    plt.suptitle("Comparación de métodos aproximados")
    # plt.show()

    # Nota:
    # - En la vida real, si ves que el histograma de pesos está MUY sesgado
    #   (unos pesos gigantes y muchos casi 0), significa que pocas muestras
    #   dominan la estimación → variancia alta.
    #
    # - Eso es típico cuando la evidencia es MUY improbable.
    #
    # - Esta observación es súper importante para entender
    #   por qué en problemas grandes necesitamos técnicas
    #   más avanzadas como MCMC (siguiente script).
