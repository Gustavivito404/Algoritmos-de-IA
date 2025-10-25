# =========================================================
# 30 - MDP PARCIALMENTE OBSERVABLE (POMDP)
# ---------------------------------------------------------
# Descripción:
#   En este módulo presentamos la versión parcialment observable
#   de un proceso de decisión de Markov:
#
#   POMDP = (S, A, P, R, Ω, O, γ)
#
#   Diferencia clave con un MDP normal:
#     - En un MDP el agente SABE exactamente en qué estado s está.
#     - En un POMDP el agente NO observa el estado real.
#       Solo recibe una OBSERVACIÓN ruidosa.
#
#   Entonces el agente no razona sobre "estoy en s",
#   sino sobre una CREENCIA (belief state) b(s):
#
#        b(s) = probabilidad de estar en cada estado s
#
#   Es decir, el "estado interno" del agente es una distribución
#   de probabilidad sobre S.
#
#   Objetivo:
#     Elegir acciones que maximizan la recompensa esperada
#     a largo plazo, dado lo que CREES del mundo.
#
#   En este script:
#     - Definimos un mini POMDP inspirado en el gridworld.
#     - Definimos creencias iniciales.
#     - Definimos observaciones ruidosas ("creo que estoy cerca de la meta").
#     - Actualizamos la creencia tras una acción y una observación.
#
#   Nota:
#     Aquí nos enfocamos en la mecánica de actualización de creencias
#     y cálculo de utilidad esperada sobre creencias, no en resolver
#     el POMDP completo (que es mucho más pesado).
#
# =========================================================

from typing import Dict, Tuple
import numpy as np

# ---------------------------------------------------------
# 1. Definimos el "mundo real" S y las acciones A
# ---------------------------------------------------------
# Usaremos una versión reducida del grid para que sea fácil ver el belief.
#
# Estados (S): posiciones posibles del robot
#   s0 = (0,0)   inicio posible
#   s1 = (0,1)
#   s2 = (0,2)   meta positiva (+10)
#
#   s3 = (1,0)
#   s4 = (1,1)   pared X (inaccesible)
#   s5 = (1,2)
#
#   s6 = (2,0)   trampa (-10)
#   s7 = (2,1)
#   s8 = (2,2)
#
# Acciones (A): ↑, ↓, ←, →
#
# Observaciones (Ω): etiquetas ruidosas:
#   "cerca_meta" , "cerca_trampa" , "desconocido"
#
# O(o | s, a): probabilidad de observar o si estás en s después de acción a.
#
# Recompensas R(s): similar al grid anterior.
#
# ---------------------------------------------------------

# Enumerar estados como índices 0..8
S = list(range(9))

# Mapa de estado -> (fila, col)
pos_estado: Dict[int, Tuple[int, int]] = {
    0:(0,0), 1:(0,1), 2:(0,2),
    3:(1,0), 4:(1,1), 5:(1,2),
    6:(2,0), 7:(2,1), 8:(2,2)
}

# Cuáles estados son pared / inválidos
paredes = {4}  # estado 4 = (1,1)

def valido(e: int) -> bool:
    return e not in paredes

# Recompensa inmediata por estar en cada estado
R: Dict[int, float] = {
    0: -1.0, 1: -1.0, 2: 10.0,
    3: -1.0, 4:  0.0, 5: -1.0,
    6:-10.0, 7: -1.0, 8: -1.0
}

acciones = ['↑','↓','←','→']

def mover_estado(e: int, a: str) -> int:
    """Transición determinista P(s'|s,a): aplica acción a estado e."""
    (i,j) = pos_estado[e]
    if a=='↑':  nxt=(i-1,j)
    elif a=='↓':nxt=(i+1,j)
    elif a=='←':nxt=(i,j-1)
    elif a=='→':nxt=(i,j+1)
    else:       nxt=(i,j)

    # buscamos si nxt corresponde a algún estado válido
    for cand, (ci,cj) in pos_estado.items():
        if (ci,cj)==nxt and valido(cand):
            return cand

    # si no podemos movernos (pared/borde), nos quedamos
    return e

# ---------------------------------------------------------
# 2. Observaciones Ω y modelo de observación O(o | s)
# ---------------------------------------------------------
# Definimos una observación parcial del entorno.
#
#   cerca_meta:
#       más probable si estás en estados próximos a la meta (2 o 5)
#
#   cerca_trampa:
#       más probable si estás en estados próximos a la trampa (6 o 7)
#
#   desconocido:
#       resto / ruido
#
# NOTA:
#   Vamos a suponer que la observación depende SOLO del estado resultante,
#   no de la acción (esto es común en muchos POMDP didácticos).
#
observaciones = ['cerca_meta','cerca_trampa','desconocido']

def prob_observacion(o: str, s: int) -> float:
    if o=='cerca_meta':
        # Alta probabilidad si el agente está en o cerca de la meta (2,5)
        return 0.8 if s in [2,5] else 0.1
    if o=='cerca_trampa':
        # Alta probabilidad si está en o cerca de la trampa (6,7)
        return 0.8 if s in [6,7] else 0.1
    if o=='desconocido':
        # Observación ambigua
        return 0.6 if s in [0,1,3,8] else 0.1
    return 0.0

# ---------------------------------------------------------
# 3. Creencia (belief state)
# ---------------------------------------------------------
# Una creencia b es un vector de |S| entradas con sum(b)=1,
# donde b[s] = P(estado real = s).
#
# Ejemplo:
#   b_inicial = distribución uniforme sobre estados válidos.
#
import numpy as np

def normalizar(v: np.ndarray) -> np.ndarray:
    s = v.sum()
    if s == 0:
        # si todo se fue a cero por alguna razón numérica,
        # devolvemos uniforme para no explotar
        return np.ones_like(v)/len(v)
    return v/s

def belief_inicial() -> np.ndarray:
    b = np.zeros(len(S))
    for s in S:
        if valido(s):
            b[s] = 1.0
    b = b / b.sum()
    return b

# ---------------------------------------------------------
# 4. Actualización de creencia en un POMDP
# ---------------------------------------------------------
# Paso típico POMDP:
#
#   - Tenemos creencia b_t sobre en qué estado estamos.
#   - Tomamos acción a.
#   - Ocurre transición P(s'|s,a).
#   - Recibimos observación o.
#   - Actualizamos creencia b_{t+1} con Bayes:
#
#   b'(s') ∝ O(o | s') * Σ_s [ P(s'|s,a) * b(s) ]
#
def predecir_creencia(b: np.ndarray, a: str) -> np.ndarray:
    """
    Predicción de la creencia según el modelo de transición.
    b_pred[s'] = Σ_s P(s'|s,a)*b[s]
    Aquí P es determinista: s' = mover_estado(s,a)
    """
    b_pred = np.zeros_like(b)
    for s in S:
        if not valido(s):
            continue
        s_prime = mover_estado(s, a)
        b_pred[s_prime] += b[s]
    return normalizar(b_pred)

def actualizar_con_observacion(b_pred: np.ndarray, o: str) -> np.ndarray:
    """
    Corrección de la creencia con la observación o:
    b_new[s'] ∝ O(o|s') * b_pred[s']
    """
    b_new = np.zeros_like(b_pred)
    for s_prime in S:
        if not valido(s_prime):
            continue
        b_new[s_prime] = prob_observacion(o, s_prime) * b_pred[s_prime]
    return normalizar(b_new)

# ---------------------------------------------------------
# 5. Utilidad esperada de una creencia
# ---------------------------------------------------------
# Si V*(s) es el valor óptimo del estado s (por ejemplo de
# iteración de valores), entonces el valor de una CREENCIA b
# es la esperanza:
#
#   V(b) = Σ_s b[s] * V*(s)
#
# Aquí, para lo didáctico, usamos V_estimada(s) = recompensa inmediata R(s)
# solo para ilustrar el cálculo.
#
def valor_de_creencia(b: np.ndarray) -> float:
    v = 0.0
    for s in S:
        if not valido(s):
            continue
        v += b[s] * R[s]
    return v

# ---------------------------------------------------------
# 6. MAIN: Simulación paso a paso
# ---------------------------------------------------------
if __name__ == "__main__":
    print("==============================================")
    print("TRACE POMDP - Creencias y Actualización Bayesiana")
    print("==============================================\n")

    # 1) Empezamos con creencia inicial (no sabemos dónde estamos exactamente)
    b = belief_inicial()
    print("Creencia inicial b(s):")
    for s in S:
        if valido(s):
            print(f"  s={s} pos={pos_estado[s]}  P={b[s]:.3f}")
    print(f"Valor esperado inicial (aprox): {valor_de_creencia(b):.3f}\n")

    # 2) Tomamos una acción, por ejemplo a = '→' (movernos a la derecha)
    accion = '→'
    b_pred = predecir_creencia(b, accion)
    print(f"Tras acción '{accion}', creencia PREDICHA b'(s):")
    for s in S:
        if valido(s):
            print(f"  s={s} pos={pos_estado[s]}  P={b_pred[s]:.3f}")
    print(f"Valor esperado tras mover (solo R inmediata): {valor_de_creencia(b_pred):.3f}\n")

    # 3) Recibimos una observación ruidosa, por ejemplo "cerca_meta"
    obs = 'cerca_meta'
    b_new = actualizar_con_observacion(b_pred, obs)
    print(f"Tras observar '{obs}', creencia ACTUALIZADA b''(s):")
    for s in S:
        if valido(s):
            print(f"  s={s} pos={pos_estado[s]}  P={b_new[s]:.3f}")
    print(f"Valor esperado posterior (aprox): {valor_de_creencia(b_new):.3f}\n")

    # Nota:
    # - En un MDP normal, el "estado" es (i,j).
    # - En un POMDP, el "estado interno" del agente es una DISTRIBUCIÓN
    #   sobre posibles (i,j).
    #
    # - El agente no pregunta "¿dónde estoy?" sino "¿con qué probabilidad
    #   estoy en cada celda?" y actúa en base a esa creencia.
    #
    # - En planificación para robótica móvil real, esto es súper común:
    #   el robot tiene incertidumbre de localización, usa sensores
    #   (observaciones ruidosas), actualiza su creencia, y decide
    #   la siguiente acción maximizando utilidad esperada futura.
