# =========================================================
# 22 - RECONOCIMIENTO DEL HABLA (HMM + VITERBI)
# ---------------------------------------------------------
# Descripción:
#   En reconocimiento de voz clásico, modelamos:
#
#   - Estados ocultos: fonemas o sub-fonemas que se están pronunciando.
#   - Observaciones: "lo que escucha" el micrófono en cada instante
#                    (por ejemplo, características acústicas).
#
#   Queremos encontrar la SECUENCIA DE FONEMAS más probable
#   que generó las observaciones medidas.
#
#   Esto es exactamente el uso de:
#       • Modelos Ocultos de Markov (HMM)
#       • Decodificación Viterbi
#
#   En este script:
#       - Definimos un mini vocabulario de fonemas:
#             ["S", "A", "L"]
#         que podrías pensar como la palabra "SAL".
#
#       - Definimos observaciones acústicas simbólicas:
#             ["sibilante", "abierta", "líquida"]
#         (esto simula vectores MFCC ya clusterizados).
#
#       - Generamos una secuencia observada y calculamos:
#             1) Filtrado online de creencias P(fonema_t | obs_1:t)
#             2) Viterbi para recuperar la SECUENCIA MÁS PROBABLE.
#
#   Nota:
#       En la vida real, las observaciones serían vectores continuos
#       (MFCC, espectro, etc.) y las palabras son HMM encadenados.
#       Aquí lo hacemos discreto y compacto para estudio.
#
# Grafo conceptual HMM (simplificado):
#
#     S  --p-->  A  --p-->  L
#     |          |          |
#     v          v          v
#   obs1       obs2       obs3
#
#   Donde cada fonema emite ciertos tipos de sonido con
#   distintas probabilidades.
#
# =========================================================

from typing import Dict, List, Tuple
import math

# ---------------------------------------------------------
# 1. Definición del modelo HMM para fonemas
# ---------------------------------------------------------
# Estados ocultos (fonemas)
estados = ["S", "A", "L"]

# Posibles observaciones acústicas categóricas
observaciones_posibles = ["sibilante", "abierta", "liquida"]

# Probabilidades iniciales P(fonema al inicio)
P_inicial = {
    "S": 0.8,
    "A": 0.2,
    "L": 0.0
}

# Probabilidades de transición entre fonemas
# P(siguiente | actual)
P_transicion = {
    "S": {"S": 0.1, "A": 0.8, "L": 0.1},
    "A": {"S": 0.0, "A": 0.2, "L": 0.8},
    "L": {"S": 0.0, "A": 0.0, "L": 1.0}
}

# Probabilidades de emisión acústica
# P(observación | fonema)
P_emision = {
    "S": {"sibilante": 0.85, "abierta": 0.10, "liquida": 0.05},
    "A": {"sibilante": 0.05, "abierta": 0.90, "liquida": 0.05},
    "L": {"sibilante": 0.05, "abierta": 0.10, "liquida": 0.85}
}

# ---------------------------------------------------------
# 2. Secuencia de observaciones que "escuchó" el micrófono
# ---------------------------------------------------------
# Imagina que el sistema de audio procesó 5 frames de voz y
# clasificó cada frame acústico en una de estas etiquetas
# discretas:
#
#   "sibilante" ~ fonema tipo S
#   "abierta"   ~ fonema tipo A
#   "liquida"   ~ fonema tipo L
#
# Observación real medida:
observaciones = ["sibilante", "abierta", "abierta", "liquida", "liquida"]

# ---------------------------------------------------------
# 3. Función de normalización para distribuciones
# ---------------------------------------------------------
def normalizar(d: Dict[str, float]) -> Dict[str, float]:
    total = sum(d.values())
    for k in d:
        d[k] /= total
    return d

# ---------------------------------------------------------
# 4. FILTRADO (tipo forward online)
#    Calcula la creencia P(estado_t | obs_1:t)
# ---------------------------------------------------------
def filtrar_secuencia(obs_seq: List[str]) -> List[Dict[str, float]]:
    # Paso inicial: combinar P_inicial con la evidencia inicial
    belief_actual = {}
    primera_obs = obs_seq[0]
    for s in estados:
        belief_actual[s] = P_inicial[s] * P_emision[s][primera_obs]
    belief_actual = normalizar(belief_actual)

    beliefs = [belief_actual]

    # Para cada observación siguiente:
    for obs in obs_seq[1:]:
        # 1) Predicción desde estados anteriores
        pred = {}
        for s_actual in estados:
            pred[s_actual] = sum(
                beliefs[-1][s_prev] * P_transicion[s_prev][s_actual]
                for s_prev in estados
            )

        # 2) Actualización con la nueva observación
        for s_actual in estados:
            pred[s_actual] *= P_emision[s_actual][obs]

        # 3) Normalizar
        pred = normalizar(pred)
        beliefs.append(pred)

    return beliefs

# ---------------------------------------------------------
# 5. VITERBI
#    Encuentra la SECUENCIA DE ESTADOS MÁS PROBABLE
#    dada TODA la secuencia de observaciones.
# ---------------------------------------------------------
def viterbi(obs_seq: List[str]) -> List[str]:
    # delta[t][estado] = log prob máx de una trayectoria que termina en 'estado' en t
    # psi[t][estado]   = estado anterior que mejor llevó aquí
    delta: List[Dict[str, float]] = []
    psi:   List[Dict[str, str]]   = []

    # Inicialización (t = 0)
    delta_t0 = {}
    psi_t0 = {}
    primera_obs = obs_seq[0]
    for s in estados:
        # usamos log para evitar underflow: log(a*b) = log(a)+log(b)
        delta_t0[s] = math.log(P_inicial[s] + 1e-12) + math.log(P_emision[s][primera_obs] + 1e-12)
        psi_t0[s] = None
    delta.append(delta_t0)
    psi.append(psi_t0)

    # Recurrencia (t = 1..T-1)
    for t in range(1, len(obs_seq)):
        obs_actual = obs_seq[t]
        delta_t = {}
        psi_t = {}
        for s in estados:
            # buscamos el mejor estado previo s_prev que maximiza:
            # delta[t-1][s_prev] + log(P_transicion[s_prev][s]) + log(P_emision[s][obs_actual])
            mejor_score = -1e18
            mejor_prev = None
            for s_prev in estados:
                score = (
                    delta[t-1][s_prev] +
                    math.log(P_transicion[s_prev][s] + 1e-12) +
                    math.log(P_emision[s][obs_actual] + 1e-12)
                )
                if score > mejor_score:
                    mejor_score = score
                    mejor_prev = s_prev
            delta_t[s] = mejor_score
            psi_t[s] = mejor_prev
        delta.append(delta_t)
        psi.append(psi_t)

    # Backtracking:
    # 1) elegimos el mejor estado final
    ultimo_t = len(obs_seq) - 1
    mejor_estado_final = max(delta[ultimo_t], key=lambda s: delta[ultimo_t][s])

    # 2) reconstruimos hacia atrás
    secuencia = [None] * len(obs_seq)
    secuencia[ultimo_t] = mejor_estado_final
    for t in range(ultimo_t, 0, -1):
        secuencia[t-1] = psi[t][secuencia[t]]

    return secuencia

# ---------------------------------------------------------
# 6. MAIN: Traza clara de resultados
# ---------------------------------------------------------
if __name__ == "__main__":
    print("==============================================")
    print("TRACE RECONOCIMIENTO DEL HABLA (HMM)")
    print("==============================================")

    print("Observaciones acústicas detectadas (frames):")
    print(" -> ", observaciones)
    print("Leyenda: 'sibilante'~S, 'abierta'~A, 'liquida'~L\n")

    # 6.1. FILTRADO: creencia frame a frame
    beliefs = filtrar_secuencia(observaciones)

    print("----------------------------------------------")
    print("FILTRADO ONLINE (creencia probabilística)")
    print("----------------------------------------------")
    for t, b in enumerate(beliefs):
        print(f"Frame {t:02d} | Obs='{observaciones[t]}' | "
              f"P(S)={b['S']:.3f}  P(A)={b['A']:.3f}  P(L)={b['L']:.3f}")

    # 6.2. VITERBI: mejor secuencia de fonemas
    secuencia_fonemas = viterbi(observaciones)

    print("\n----------------------------------------------")
    print("VITERBI (secuencia de fonemas más probable)")
    print("----------------------------------------------")
    print("Frames:           ", [f"{i:02d}" for i in range(len(observaciones))])
    print("Observación mic:  ", observaciones)
    print("Fonema estimado:  ", secuencia_fonemas)

    # 6.3. Interpretación final
    print("\n==============================================")
    print("RESULTADO FINAL")
    print("==============================================")
    print("Secuencia estimada de fonemas pronunciados:")
    print("  ", " - ".join(secuencia_fonemas))
    print("\nNota:")
    print("  - 'S' suena tipo consonante sibilante (ssss)")
    print("  - 'A' suena vocal abierta tipo 'aaa'")
    print("  - 'L' suena consonante líquida tipo 'lll'")
    print("  El sistema concluye qué fonema se estaba diciendo")
    print("  en cada frame de audio según el HMM y Viterbi.")