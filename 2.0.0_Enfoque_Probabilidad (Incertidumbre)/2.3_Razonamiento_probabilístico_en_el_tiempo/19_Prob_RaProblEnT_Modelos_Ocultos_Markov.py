# =========================================================
# 19 - MODELOS OCULTOS DE MARKOV (HMM)
# ---------------------------------------------------------
# Descripción:
#   Un Modelo Oculto de Markov (HMM = Hidden Markov Model)
#   describe dos cosas:
#
#   1) Un proceso de estados OCULTOS que sigue una cadena de Markov.
#      Ejemplo aquí:
#          Estado_t ∈ {Lluvia, Soleado}
#      con transiciones P(Estado_t+1 | Estado_t).
#
#   2) Un proceso de OBSERVACIÓN donde cada estado oculto genera
#      una evidencia visible con cierta probabilidad.
#      Ejemplo aquí:
#          Evidencia_t ∈ {Mojado, Seco}
#      con probabilidades P(Evidencia_t | Estado_t).
#
#   El punto clave:
#      - No vemos directamente si llueve.
#      - Solo vemos si el suelo está mojado.
#      - A partir de las observaciones queremos inferir
#        las probabilidades sobre los estados ocultos.
#
#   Este script hace 3 cosas:
#     A) Genera una secuencia sintética (estado oculto + evidencia).
#     B) Hace FILTRADO paso a paso (forward online).
#     C) Hace VITERBI para obtener la secuencia OCULTA más probable.
#
# Características:
#   - Filtrado = creencia probabilística P(Estado_t | evidencias hasta t)
#   - Viterbi = mejor ruta única más probable (decodificación MAP secuencial)
#
# Nota:
#   Viterbi ≈ "¿qué secuencia de clima es más probable que produjo estas observaciones?".
# =========================================================

import random
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Definimos el HMM
# ---------------------------------------------------------
estados_ocultos = ["Lluvia", "Soleado"]
observables = ["Mojado", "Seco"]

# Matriz de transición P(X_t+1 | X_t)
P_transicion = {
    "Lluvia":   {"Lluvia": 0.7, "Soleado": 0.3},
    "Soleado":  {"Lluvia": 0.4, "Soleado": 0.6}
}

# Modelo sensor / emisión P(Evidencia | Estado)
P_emision = {
    "Lluvia":   {"Mojado": 0.9, "Seco": 0.1},
    "Soleado":  {"Mojado": 0.2, "Seco": 0.8}
}

# Distribución inicial P(X_0)
P_inicial = {"Lluvia": 0.5, "Soleado": 0.5}

# ---------------------------------------------------------
# 2. Muestrear una secuencia (oculto + observado)
# ---------------------------------------------------------
def sample_estado_siguiente(estado_actual: str) -> str:
    r = random.random()
    p_lluvia = P_transicion[estado_actual]["Lluvia"]
    if r < p_lluvia:
        return "Lluvia"
    else:
        return "Soleado"

def sample_observacion(estado_actual: str) -> str:
    r = random.random()
    p_mojado = P_emision[estado_actual]["Mojado"]
    if r < p_mojado:
        return "Mojado"
    else:
        return "Seco"

def generar_secuencia(longitud: int, semilla: int = 7) -> Tuple[List[str], List[str]]:
    """
    Genera una trayectoria de estados ocultos reales y observaciones visibles.
    """
    random.seed(semilla)

    # Elegir estado inicial según P_inicial
    r0 = random.random()
    if r0 < P_inicial["Lluvia"]:
        estado = "Lluvia"
    else:
        estado = "Soleado"

    ocultos = [estado]
    observaciones = [sample_observacion(estado)]

    for _ in range(1, longitud):
        estado = sample_estado_siguiente(ocultos[-1])
        ocultos.append(estado)
        observaciones.append(sample_observacion(estado))

    return ocultos, observaciones

# ---------------------------------------------------------
# 3. Filtrado (Forward Online)
# ---------------------------------------------------------
def normalizar(dist: Dict[str, float]) -> Dict[str,float]:
    total = sum(dist.values())
    for k in dist:
        dist[k] /= total
    return dist

def paso_forward(prev_belief: Dict[str,float], evidencia_actual: str) -> Dict[str,float]:
    """
    Un paso de filtrado:
      1) predecir P(X_t) a partir de P(X_{t-1})
      2) incorporar evidencia actual
      3) normalizar
    """
    # Predicción por transición
    pred = {}
    for x in estados_ocultos:
        pred[x] = sum(prev_belief[x_prev] * P_transicion[x_prev][x] for x_prev in estados_ocultos)

    # Actualización con evidencia
    for x in estados_ocultos:
        pred[x] *= P_emision[x][evidencia_actual]

    # Normalizar
    return normalizar(pred)

def filtrar_secuencia(obs_seq: List[str]) -> List[Dict[str,float]]:
    """
    Devuelve una lista de creencias filtradas:
    belief[t] = P(X_t|e_1...e_t)
    """
    beliefs = []
    belief_actual = P_inicial.copy()
    # incorporar la primera observación correctamente
    first_obs = obs_seq[0]
    for x in estados_ocultos:
        belief_actual[x] *= P_emision[x][first_obs]
    belief_actual = normalizar(belief_actual)
    beliefs.append(belief_actual)

    for evidencia in obs_seq[1:]:
        belief_actual = paso_forward(belief_actual, evidencia)
        beliefs.append(belief_actual)

    return beliefs

# ---------------------------------------------------------
# 4. Algoritmo de Viterbi
# ---------------------------------------------------------
def viterbi(obs_seq: List[str]) -> List[str]:
    """
    Encuentra la SECUENCIA DE ESTADOS MÁS PROBABLE que generó
    las observaciones dadas.
    Devuelve una lista de estados ocultos (decodificación MAP de la trayectoria).
    """

    # delta[t][x] = probabilidad más alta de cualquier secuencia que termine en estado x en el tiempo t
    # psi[t][x]   = estado anterior que maximizó delta
    delta = []
    psi = []

    # Inicialización t=0
    delta_t0 = {}
    psi_t0 = {}
    for x in estados_ocultos:
        delta_t0[x] = P_inicial[x] * P_emision[x][obs_seq[0]]
        psi_t0[x] = None
    # normalizamos para estabilidad numérica (opcional)
    s0 = sum(delta_t0.values())
    for x in delta_t0:
        delta_t0[x] /= s0
    delta.append(delta_t0)
    psi.append(psi_t0)

    # Recurrencia t=1..T-1
    for t in range(1, len(obs_seq)):
        delta_t = {}
        psi_t = {}
        for x in estados_ocultos:
            # buscamos el mejor estado anterior que lleva a x
            mejor_estado_anterior = None
            mejor_valor = -1.0
            for x_prev in estados_ocultos:
                valor_candidato = (
                    delta[t-1][x_prev] *
                    P_transicion[x_prev][x] *
                    P_emision[x][obs_seq[t]]
                )
                if valor_candidato > mejor_valor:
                    mejor_valor = valor_candidato
                    mejor_estado_anterior = x_prev

            delta_t[x] = mejor_valor
            psi_t[x] = mejor_estado_anterior

        # normalizar para estabilidad
        st = sum(delta_t.values())
        for x in delta_t:
            delta_t[x] /= st

        delta.append(delta_t)
        psi.append(psi_t)

    # Backtracking: elegir el mejor estado final y retroceder
    # Paso 1: mejor estado al final
    ultimo_t = len(obs_seq) - 1
    mejor_estado_final = max(delta[ultimo_t], key=lambda x: delta[ultimo_t][x])

    secuencia_estimada = [None] * len(obs_seq)
    secuencia_estimada[ultimo_t] = mejor_estado_final

    # Paso 2: ir hacia atrás
    for t in range(ultimo_t, 0, -1):
        secuencia_estimada[t-1] = psi[t][secuencia_estimada[t]]

    return secuencia_estimada

# ---------------------------------------------------------
# 5. MAIN (traza completa)
# ---------------------------------------------------------
if __name__ == "__main__":
    # Generamos una secuencia sintética
    ocultos_reales, observaciones = generar_secuencia(longitud=10, semilla=42)

    print("==============================================")
    print("SECUENCIA GENERADA (HMM)")
    print("==============================================")
    print("Día | Estado real | Observación")
    print("----+-------------+------------")
    for i, (st, obs) in enumerate(zip(ocultos_reales, observaciones), start=1):
        print(f"{i:3d} | {st:<11} | {obs}")

    # Filtrado paso a paso
    beliefs = filtrar_secuencia(observaciones)

    print("\n==============================================")
    print("FILTRADO (creencia online)")
    print("P(Estado_t | observaciones hasta t)")
    print("==============================================")
    for i, b in enumerate(beliefs, start=1):
        print(f"Día {i:2d} | P(Lluvia)={b['Lluvia']:.3f} | P(Soleado)={b['Soleado']:.3f} | Obs={observaciones[i-1]}")

    # Viterbi (secuencia más probable)
    secuencia_estimada = viterbi(observaciones)

    print("\n==============================================")
    print("DECODIFICACIÓN VITERBI")
    print("Secuencia oculta más probable dada toda la evidencia")
    print("==============================================")
    print("Día | Est_Real    | Est_Viterbi | Obs")
    print("----+-------------+-------------+-------")
    for i, (real, est, obs) in enumerate(zip(ocultos_reales, secuencia_estimada, observaciones), start=1):
        print(f"{i:3d} | {real:<11} | {est:<11} | {obs}")

    # -----------------------------------------------------
    # 6. Visualización opcional
    # -----------------------------------------------------
    # Podemos mapear:
    #   Lluvia   -> 1
    #   Soleado  -> 0
    #   Mojado   -> 1
    #   Seco     -> 0
    #
    # y graficar tres líneas:
    #   - Estado real oculto
    #   - Estado estimado por Viterbi
    #   - Observación (mojado/seco)
    #
    # Esto ayuda mucho a "ver" el HMM.
    
    cod = {"Lluvia":1, "Soleado":0, "Mojado":1, "Seco":0}
    
    y_real = [cod[s] for s in ocultos_reales]
    y_vit  = [cod[s] for s in secuencia_estimada]
    y_obs  = [cod[o] for o in observaciones]
    
    plt.figure(figsize=(8,4))
    plt.step(range(1,len(y_real)+1), y_real, where="post", label="Estado real", linewidth=2)
    plt.step(range(1,len(y_vit)+1), y_vit,  where="post", label="Viterbi", linestyle="--")
    plt.step(range(1,len(y_obs)+1), y_obs,  where="post", label="Observación (mojado=1)", linestyle=":")
    plt.ylim(-0.2,1.2)
    plt.yticks([0,1],["Soleado/Seco","Lluvia/Mojado"])
    plt.xlabel("Día")
    plt.ylabel("Valor binario")
    plt.title("HMM: estado oculto vs estimación vs sensor")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.show()
