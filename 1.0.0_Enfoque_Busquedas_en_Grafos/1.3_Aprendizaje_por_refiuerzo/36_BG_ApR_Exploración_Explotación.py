# =========================================================
# 36 - EXPLORACIÓN vs. EXPLOTACIÓN
# ---------------------------------------------------------
# Descripción:
#   Simulación de un agente que aprende con diferentes
#   niveles de exploración ε en un grid simple (3x3).
#
#   Observamos cómo cambia el aprendizaje y las políticas.
# =========================================================

import numpy as np
import random

# ---------------------------------------------------------
# 1. Entorno tipo gridworld (mismo que en Q-Learning)
# ---------------------------------------------------------
ACCIONES = ["↑","↓","←","→"]
GRID_SIZE = (3,3)

def mover(pos, accion):
    i,j = pos
    if accion == "↑": i = max(i-1,0)
    elif accion == "↓": i = min(i+1,GRID_SIZE[0]-1)
    elif accion == "←": j = max(j-1,0)
    elif accion == "→": j = min(j+1,GRID_SIZE[1]-1)
    return (i,j)

def recompensa(pos):
    if pos == (2,2): return 10
    if pos == (2,0): return -10
    return -1

def es_terminal(pos):
    return pos in [(2,2),(2,0)]

# ---------------------------------------------------------
# 2. Q-Learning encapsulado en función
# ---------------------------------------------------------
def entrenar(epsilon, episodios=30):
    alpha, gamma = 0.2, 0.9
    Q = {}
    for ep in range(episodios):
        pos = (0,0)
        while not es_terminal(pos):
            if random.random() < epsilon:
                a = random.choice(ACCIONES)
            else:
                vals = [Q.get((pos,a),0) for a in ACCIONES]
                a = random.choice([act for act,v in zip(ACCIONES,vals) if v == max(vals)])
            nueva = mover(pos, a)
            r = recompensa(nueva)
            max_q_siguiente = max(Q.get((nueva,a2),0) for a2 in ACCIONES)
            Q[(pos,a)] = Q.get((pos,a),0) + alpha*(r + gamma*max_q_siguiente - Q.get((pos,a),0))
            pos = nueva
    return Q

# ---------------------------------------------------------
# 3. Construcción de política a partir de Q
# ---------------------------------------------------------
def politica_greedy(Q):
    politica = np.empty(GRID_SIZE, dtype='<U2')
    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            s = (i,j)
            if es_terminal(s):
                politica[i,j] = "X"
            else:
                valores = [Q.get((s,a),0) for a in ACCIONES]
                max_v = max(valores)
                mejores = [a for a,v in zip(ACCIONES,valores) if v==max_v]
                politica[i,j] = random.choice(mejores)
    return politica

# ---------------------------------------------------------
# 4. Simulación con tres niveles de exploración
# ---------------------------------------------------------
niveles = [0.0, 0.2, 0.8]  # sin exploración, moderada, alta
politicas = {}

print("==============================================")
print("TRACE EXPLORACIÓN vs. EXPLOTACIÓN")
print("==============================================\n")

for eps in niveles:
    print(f">>> ENTRENANDO con ε = {eps}")
    Q = entrenar(eps)
    politicas[eps] = politica_greedy(Q)
    print("POLÍTICA APRENDIDA:")
    for i in range(GRID_SIZE[0]):
        fila = " ".join(f"{politicas[eps][i,j]:>3s}" for j in range(GRID_SIZE[1]))
        print(f"Fila {i}: {fila}")
    print("")

# Nota:
# - ε=0.0  => el agente nunca explora (puede atascarse).
# - ε=0.2  => buen equilibrio, explora un poco.
# - ε=0.8  => demasiado ruido, no converge fácilmente.
