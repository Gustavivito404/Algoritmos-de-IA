# =========================================================
# 27 - MODELOS DE MARKOV OCULTOS (HMM)
# ---------------------------------------------------------
# Descripción:
#   Un Modelo de Markov Oculto (HMM) describe:
#
#   - Estados ocultos (no observados directamente)
#       Ej: Clima_t ∈ {Lluvia, Soleado}
#
#   - Observaciones visibles generadas por esos estados
#       Ej: Suelo_t ∈ {Mojado, Seco}
#
#   Asumimos:
#       1) El estado oculto sigue una cadena de Markov:
#              P(X_t | X_{t-1})
#
#       2) Cada observación depende SOLO del estado actual:
#              P(E_t | X_t)
#
#   Problema típico:
#       Dada una secuencia de observaciones (ej. Mojado/Seco),
#       ¿cuál es la SECUENCIA DE ESTADOS OCULTOS más probable?
#
#   Solución:
#       Algoritmo de Viterbi.
#
# =========================================================

from typing import Dict, List
import math

# ---------------------------------------------------------
# 1. Definimos el HMM
# ---------------------------------------------------------
estados = ["Lluvia", "Soleado"]
observaciones_posibles = ["Mojado", "Seco"]

# Probabilidad inicial P(X_0)
P_inicial = {
    "Lluvia": 0.5,
    "Soleado": 0.5
}

# Transición P(X_t | X_{t-1})
P_transicion = {
    "Lluvia":   {"Lluvia": 0.7, "Soleado": 0.3},
    "Soleado":  {"Lluvia": 0.4, "Soleado": 0.6}
}

# Emisión / sensor P(Obs | Estado)
P_emision = {
    "Lluvia":   {"Mojado": 0.9, "Seco": 0.1},
    "Soleado":  {"Mojado": 0.2, "Seco": 0.8}
}

# ---------------------------------------------------------
# 2. Evidencia observada
# ---------------------------------------------------------
# Ejemplo: el suelo estuvo así durante 6 días:
observaciones = ["Mojado", "Mojado", "Seco", "Mojado", "Mojado", "Seco"]

# ---------------------------------------------------------
# 3. Viterbi
# ---------------------------------------------------------
def viterbi(obs_seq: List[str]) -> List[str]:
    """
    Devuelve la secuencia de estados ocultos más probable:
    argmax_x P(x_1...x_T | obs_1...obs_T)
    usando programación dinámica con log-probabilidades.
    """
    # delta[t][estado] = log prob máx de una trayectoria que termina en 'estado' en t
    # psi[t][estado]   = mejor estado anterior que lleva ahí
    delta: List[Dict[str, float]] = []
    psi:   List[Dict[str, str]]   = []

    # Inicialización (t=0)
    delta_t0 = {}
    psi_t0 = {}
    primera_obs = obs_seq[0]
    for s in estados:
        # log P(X0=s) + log P(obs0 | s)
        delta_t0[s] = math.log(P_inicial[s] + 1e-12) + math.log(P_emision[s][primera_obs] + 1e-12)
        psi_t0[s] = None
    delta.append(delta_t0)
    psi.append(psi_t0)

    # Recurrencia (t=1..T-1)
    for t in range(1, len(obs_seq)):
        obs_actual = obs_seq[t]
        delta_t = {}
        psi_t = {}
        for s in estados:
            mejor_score = -1e18
            mejor_prev = None
            for s_prev in estados:
                # score candidato:
                #   delta[t-1][s_prev] + log P(s|s_prev) + log P(obs_actual|s)
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

    # Terminación: mejor estado final
    ultimo_t = len(obs_seq) - 1
    mejor_estado_final = max(delta[ultimo_t], key=lambda s: delta[ultimo_t][s])

    # Backtracking (reconstruir ruta)
    secuencia_estimada = [None] * len(obs_seq)
    secuencia_estimada[ultimo_t] = mejor_estado_final
    for t in range(ultimo_t, 0, -1):
        secuencia_estimada[t-1] = psi[t][secuencia_estimada[t]]

    return secuencia_estimada

# ---------------------------------------------------------
# 4. Filtrado (creencia por día)
# ---------------------------------------------------------
def normalizar(d: Dict[str, float]) -> Dict[str, float]:
    total = sum(d.values())
    for k in d:
        d[k] /= total
    return d

def filtrar_secuencia(obs_seq: List[str]) -> List[Dict[str, float]]:
    """
    belief[t] = P(estado_t | obs_1...obs_t)
    Esto es el filtrado 'forward' online.
    """
    beliefs = []

    # t = 0 (inicial con primera observación)
    b0 = {}
    o0 = obs_seq[0]
    for s in estados:
        b0[s] = P_inicial[s] * P_emision[s][o0]
    b0 = normalizar(b0)
    beliefs.append(b0)

    # t > 0
    for obs in obs_seq[1:]:
        pred = {}
        # predicción P(X_t=s) = sum_s' P(X_{t-1}=s') * P(s|s')
        for s in estados:
            pred[s] = sum(
                beliefs[-1][s_prev] * P_transicion[s_prev][s]
                for s_prev in estados
            )
        # actualización con la nueva observación
        for s in estados:
            pred[s] *= P_emision[s][obs]
        pred = normalizar(pred)
        beliefs.append(pred)

    return beliefs

# ---------------------------------------------------------
# 5. MAIN - Trazas y salida
# ---------------------------------------------------------
if __name__ == "__main__":
    print("==============================================")
    print("TRACE HMM - CLIMA OCULTO")
    print("==============================================")
    print("Observaciones (lo que vemos):")
    print(" ", observaciones)
    print("Significado: 'Mojado' ~ suelo húmedo, 'Seco' ~ suelo seco")
    print("Los estados ocultos posibles son: 'Lluvia' o 'Soleado'\n")

    # Paso 1: Filtrado online (creencia día a día)
    beliefs = filtrar_secuencia(observaciones)

    print("----------------------------------------------")
    print("FILTRADO ONLINE (creencia probabilística)")
    print("----------------------------------------------")
    for t, b in enumerate(beliefs):
        print(f"Día {t:02d} | Obs={observaciones[t]:7s} | "
              f"P(Lluvia)={b['Lluvia']:.3f}  P(Soleado)={b['Soleado']:.3f}")

    # Paso 2: Viterbi (secuencia más probable completa)
    secuencia_estimada = viterbi(observaciones)

    print("\n----------------------------------------------")
    print("VITERBI (secuencia oculta más probable)")
    print("----------------------------------------------")
    print("Día:             ", [f"{i:02d}" for i in range(len(observaciones))])
    print("Observación:     ", observaciones)
    print("Clima estimado:  ", secuencia_estimada)

    print("\n==============================================")
    print("RESULTADO FINAL")
    print("==============================================")
    print("Secuencia estimada de clima (oculto):")
    print("  ", " -> ".join(secuencia_estimada))
    print("\nComentario:")
    print("  - Filtrado te da: prob de Lluvia/Soleado en cada día.")
    print("  - Viterbi te da UNA sola explicación global más probable.")
    print("  - Esto es la base clásica del reconocimiento de voz,")
    print("    etiquetado de secuencias y seguimiento temporal.")
