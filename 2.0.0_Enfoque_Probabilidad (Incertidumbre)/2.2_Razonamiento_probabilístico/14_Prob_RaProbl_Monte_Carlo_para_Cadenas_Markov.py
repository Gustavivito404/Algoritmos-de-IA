# =========================================================
# 14 - MONTE CARLO PARA CADENAS DE MARKOV (MCMC - GIBBS)
# ---------------------------------------------------------
# Descripción:
#   En este script implementamos una forma sencilla de
#   muestreo MCMC (Monte Carlo Markov Chain), usando
#   Gibbs Sampling en nuestra Red Bayesiana:
#
#       Lluvia ──► SensorAgua ──► Alarma
#
#   Objetivo:
#     Estimar P(Lluvia | Alarma=True)
#     generando una CADENA de estados que lentamente
#     se acerca a la distribución correcta.
#
# ¿Qué es Gibbs Sampling (versión intuitiva)?
#   - Imagina que tienes todas tus variables.
#   - Fijas la evidencia (por ejemplo Alarma=True).
#   - Para las variables ocultas, las "resampleas" una por una,
#     usando su distribución condicional dada las demás.
#
#   Eso crea una cadena de estados:
#       estado_1 -> estado_2 -> estado_3 -> ...
#   y la frecuencia con la que aparezca cada valor
#   aproxima la distribución posterior que queremos.
#
# Características:
#   - NO rechazamos muestras.
#   - NO necesitamos calcular explícitamente la prob conjunta completa cada vez.
#   - Funciona incluso con evidencia rara.
#
# Este es un método FUNDAMENTAL en inferencia bayesiana grande.
#
# Matemática base (resumen):
#   Para muestrear una variable X en Gibbs:
#      P(X = x | resto, evidencia)
#   se calcula usando los factores locales relevantes
#   (Manto de Markov de X).
#
#   En esta red:
#      MantoDeMarkov(Lluvia) = {SensorAgua}
#      MantoDeMarkov(SensorAgua) = {Lluvia, Alarma}
#      (Alarma es evidencia fija en este ejemplo)
#
# ---------------------------------------------------------
# Ejemplo:
#   Estimamos P(Lluvia | Alarma=True)
#   corriendo Gibbs Sampling muchas iteraciones y contando
#   cuántas veces Lluvia=True.
# =========================================================

import random
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

# ---------------------------------------------------------
# 1. Probabilidades de la Red Bayesiana
# ---------------------------------------------------------
P_Lluvia = {True: 0.3, False: 0.7}

# P(SensorAgua | Lluvia)
P_SensorAgua_dado_Lluvia = {
    (True,  True): 0.9,
    (False, True): 0.1,
    (True,  False): 0.2,
    (False, False): 0.8
}

# P(Alarma | SensorAgua)
P_Alarma_dado_Sensor = {
    (True,  True): 0.95,
    (False, True): 0.05,
    (True,  False): 0.01,
    (False, False): 0.99
}

# ---------------------------------------------------------
# 2. Funciones auxiliares de probabilidad condicional local
# ---------------------------------------------------------
def p_sensor_dado_lluvia(sensor_val: bool, lluvia_val: bool) -> float:
    """P(SensorAgua = sensor_val | Lluvia = lluvia_val)."""
    if sensor_val:
        return P_SensorAgua_dado_Lluvia[(True, lluvia_val)]
    else:
        return P_SensorAgua_dado_Lluvia[(False, lluvia_val)]

def p_lluvia(lluvia_val: bool) -> float:
    """P(Lluvia = lluvia_val). Nodo raíz."""
    return P_Lluvia[lluvia_val]

def p_alarma_dado_sensor(alarma_val: bool, sensor_val: bool) -> float:
    """P(Alarma = alarma_val | SensorAgua = sensor_val)."""
    if alarma_val:
        return P_Alarma_dado_Sensor[(True, sensor_val)]
    else:
        return P_Alarma_dado_Sensor[(False, sensor_val)]

# ---------------------------------------------------------
# 3. Distribuciones condicionales necesarias para Gibbs
# ---------------------------------------------------------
# Gibbs re-samplea una variable condicionada en su "Manto de Markov".
#
#   Para Lluvia:
#       Manto(Lluvia) = {SensorAgua}
#   Entonces:
#       P(Lluvia | SensorAgua = s)
#     ∝ P(Lluvia) * P(SensorAgua = s | Lluvia)
#
def sample_lluvia_condicional(sensor_val: bool) -> bool:
    # Calculamos la probabilidad no normalizada para Lluvia=True y False.
    score_true = p_lluvia(True)  * p_sensor_dado_lluvia(sensor_val, True)
    score_false = p_lluvia(False) * p_sensor_dado_lluvia(sensor_val, False)

    # Normalizamos:
    total = score_true + score_false
    p_true = score_true / total

    # Muestreamos:
    return random.random() < p_true

#   Para SensorAgua:
#       Manto(SensorAgua) = {Lluvia, Alarma}
#   Entonces:
#       P(SensorAgua | Lluvia, Alarma=True)
#     ∝ P(SensorAgua | Lluvia) * P(Alarma=True | SensorAgua)
#
def sample_sensor_condicional(lluvia_val: bool, alarma_es_true: bool=True) -> bool:
    # score para SensorAgua=True
    score_true = (
        p_sensor_dado_lluvia(True, lluvia_val) *
        p_alarma_dado_sensor(alarma_es_true, True)
    )
    # score para SensorAgua=False
    score_false = (
        p_sensor_dado_lluvia(False, lluvia_val) *
        p_alarma_dado_sensor(alarma_es_true, False)
    )

    total = score_true + score_false
    p_true = score_true / total

    return random.random() < p_true

# Nota:
#   No re-sampleamos "Alarma" porque en esta inferencia
#   la tratamos como evidencia fija: Alarma=True.

# ---------------------------------------------------------
# 4. Gibbs Sampling
# ---------------------------------------------------------
def gibbs_mcmc(num_iteraciones: int,
               evidencia: Dict[str,bool],
               burn_in: int = 1000) -> Dict[str, float]:
    """
    Ejecuta Gibbs Sampling sobre las variables ocultas.
    - evidencia: dict con valores fijos (por ejemplo {"Alarma": True})
    - burn_in: cuántas primeras muestras descartamos para
               dejar que la cadena "se estabilice"
    Devuelve:
        frecuencias estimadas de cada valor de Lluvia.
    """

    # 1) Inicializamos un estado consistente con la evidencia.
    #    Elegimos valores iniciales (pueden ser aleatorios).
    estado = {}

    # Fijamos Alarma desde la evidencia
    estado["Alarma"] = evidencia["Alarma"]

    # Inicializamos Lluvia aleatoriamente según su prior
    estado["Lluvia"] = random.random() < P_Lluvia[True]

    # Inicializamos SensorAgua aleatoriamente según esa Lluvia,
    # pero también consistente con que Alarma es True.
    # (Para inicializar podemos ignorar consistencia estricta,
    #  esto sólo da un punto de arranque. Luego Gibbs corrige.)
    estado["SensorAgua"] = random.random() < 0.5

    # Contadores para estimar la distribución posterior
    contador_lluvia_true = 0
    contador_lluvia_total = 0

    print("==============================================")
    print("TRACE MONTE CARLO MCMC (GIBBS SAMPLING)")
    print("==============================================\n")

    print("1) Estado inicial de la cadena:")
    print(f"   {estado}\n")

    # 2) Iteraciones de Gibbs:
    for it in range(num_iteraciones):
        # Re-samplear Lluvia dado SensorAgua (manto de Lluvia)
        estado["Lluvia"] = sample_lluvia_condicional(
            sensor_val=estado["SensorAgua"]
        )

        # Re-samplear SensorAgua dado Lluvia y Alarma=True
        estado["SensorAgua"] = sample_sensor_condicional(
            lluvia_val=estado["Lluvia"],
            alarma_es_true=estado["Alarma"]
        )

        # No tocamos Alarma porque es evidencia fija
        # estado["Alarma"] = True

        # Vamos guardando muestras DESPUÉS del burn-in
        if it >= burn_in:
            contador_lluvia_total += 1
            if estado["Lluvia"] is True:
                contador_lluvia_true += 1

        # Imprimimos solo algunas iteraciones iniciales para traza
        if it < 10 or it in [burn_in, burn_in+1, num_iteraciones-1]:
            print(f"[Iter {it}] Estado = {estado}")

    # Probabilidad estimada posterior
    if contador_lluvia_total == 0:
        posterior_lluvia = 0.0
    else:
        posterior_lluvia = contador_lluvia_true / contador_lluvia_total

    resultado = {
        "P(Lluvia=True | Alarma=True)": posterior_lluvia,
        "muestras_usadas": contador_lluvia_total
    }
    return resultado

# ---------------------------------------------------------
# 5. MAIN: correr Gibbs y mostrar resultados
# ---------------------------------------------------------
if __name__ == "__main__":
    random.seed(7)

    evidencia = {"Alarma": True}
    num_iter = 5000
    burn_in = 1000

    resultado = gibbs_mcmc(num_iteraciones=num_iter,
                           evidencia=evidencia,
                           burn_in=burn_in)

    print("\n2) RESULTADO FINAL MCMC:")
    print(f"   Estimación P(Lluvia=True | Alarma=True) ≈ {resultado['P(Lluvia=True | Alarma=True)']:.4f}")
    print(f"   Muestras efectivas (después de burn-in): {resultado['muestras_usadas']}")
    print("")
    print("Interpretación:")
    print("- Gibbs va ajustando las variables internas una por una,")
    print("  condicionadas en su manto de Markov.")
    print("- La cadena va 'caminando' por estados plausibles dados")
    print("  que Alarma=True.")
    print("- La fracción de estados donde Lluvia=True aproxima la")
    print("  probabilidad posterior que queremos (~0.66).")
    print("")

    # ---------------------------------------------------------
    # 6. (OPCIONAL) Curva de convergencia
    # ---------------------------------------------------------
    # Aquí podríamos guardar la estimación acumulada de
    # P(Lluvia=True | Alarma=True) conforme avanza la cadena
    # e ir graficando cómo se estabiliza.
    #
    # Para mantener el script simple y directo no lo recolectamos
    # en gibbs_mcmc(), pero si quieres analizar convergencia,
    # podemos modificar gibbs_mcmc para devolver una lista del
    # valor acumulado por iteración y luego graficar algo tipo:
    #
    # plt.plot(historial_iteraciones, estimacion_parcial)
    # plt.xlabel("Iteración")
    # plt.ylabel("Estimación P(Lluvia|Alarma)")
    # plt.title("Convergencia MCMC (Gibbs)")
    # plt.ylim(0,1)
    # plt.show()
    #
    # Esta gráfica es MUY útil cuando presentas MCMC,
    # porque enseña visualmente cómo el muestreo converge.
