# =========================================================
# 34 - APRENDIZAJE POR REFUERZO ACTIVO
# ---------------------------------------------------------
# Descripción:
#   El agente elige acciones y aprende su propia política
#   mediante exploración ε-greedy y actualización Q(s,a).
#
# =========================================================

import numpy as np
import random

# ---------------------------------------------------------
# 1. Definición del entorno (mismo grid 3x3 que antes)
# ---------------------------------------------------------
acciones = ["↑","↓","←","→"]
grid_size = (3,3)

def mover(pos, accion):
    i,j = pos
    if accion == "↑": i = max(i-1,0)
    elif accion == "↓": i = min(i+1,grid_size[0]-1)
    elif accion == "←": j = max(j-1,0)
    elif accion == "→": j = min(j+1,grid_size[1]-1)
    return (i,j)

def recompensa(pos):
    if pos == (2,2): return 10
    if pos == (2,0): return -10
    return -1

# ---------------------------------------------------------
# 2. Parámetros de aprendizaje
# ---------------------------------------------------------
alpha = 0.2      # tasa de aprendizaje
gamma = 0.9      # descuento
epsilon = 0.2    # exploración
Q = {}           # diccionario Q[(s,a)]

# ---------------------------------------------------------
# 3. Funciones auxiliares
# ---------------------------------------------------------
def mejor_accion(pos):
    """Devuelve la acción con mejor Q(s,a) conocida."""
    valores = [Q.get((pos,a),0) for a in acciones]
    max_v = max(valores)
    mejores = [a for a,v in zip(acciones,valores) if v==max_v]
    return random.choice(mejores)

def elegir_accion(pos):
    """Política ε-greedy."""
    if random.random() < epsilon:
        return random.choice(acciones)
    else:
        return mejor_accion(pos)

# ---------------------------------------------------------
# 4. Entrenamiento
# ---------------------------------------------------------
episodios = 100
for ep in range(episodios):
    pos = (0,0)
    while pos not in [(2,2),(2,0)]:
        a = elegir_accion(pos)
        nueva_pos = mover(pos, a)
        r = recompensa(nueva_pos)
        max_q_siguiente = max(Q.get((nueva_pos,a2),0) for a2 in acciones)
        Q[(pos,a)] = Q.get((pos,a),0) + alpha * (r + gamma*max_q_siguiente - Q.get((pos,a),0))
        pos = nueva_pos

# ---------------------------------------------------------
# 5. Construcción de la política aprendida
# ---------------------------------------------------------
politica = np.empty(grid_size, dtype='<U2')
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        pos = (i,j)
        if pos in [(2,2),(2,0)]:
            politica[i,j] = "X"
        else:
            politica[i,j] = mejor_accion(pos)

# ---------------------------------------------------------
# 6. Mostrar resultados
# ---------------------------------------------------------
print("==============================================")
print("TRACE APRENDIZAJE POR REFUERZO ACTIVO")
print("==============================================\n")

for (pos, acc) in sorted(Q.keys()):
    print(f"Q{pos, acc} = {Q[pos, acc]:.2f}")

print("\nPOLÍTICA APRENDIDA:")
for i in range(grid_size[0]):
    fila = " ".join(f"{politica[i,j]:>3s}" for j in range(grid_size[1]))
    print(f"Fila {i}: {fila}")

# Nota:
# - El agente ahora elige acciones y mejora su política.
# - La exploración ε evita que se estanque en un óptimo local.
# - Este método es base del Q-Learning (donde la política y Q se aprenden simultáneamente).
