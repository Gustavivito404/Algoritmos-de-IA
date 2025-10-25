# =========================================================
# 33 - APRENDIZAJE POR REFUERZO PASIVO
# ---------------------------------------------------------
# Descripción:
#   El agente sigue una política fija y aprende los valores
#   esperados de los estados (V(s)) usando aprendizaje temporal (TD).
#
# =========================================================

import numpy as np
import random

# ---------------------------------------------------------
# 1. Definimos el entorno (tipo gridworld)
# ---------------------------------------------------------
# Estados: 3x3
# Acciones: ↑, ↓, ←, →
# Recompensas:
#   +10 en la meta (2,2)
#   -10 en la trampa (2,0)
#   -1 en los demás
# Política fija: moverse hacia la derecha siempre que pueda
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

def politica_fija(pos):
    """Política simple: moverse hacia la derecha si se puede, sino hacia abajo."""
    i,j = pos
    if j < grid_size[1]-1:
        return "→"
    else:
        return "↓"

# ---------------------------------------------------------
# 2. Aprendizaje TD(0)
# ---------------------------------------------------------
alpha = 0.2      # tasa de aprendizaje
gamma = 0.9      # descuento
V = np.zeros(grid_size)  # valores iniciales

# ---------------------------------------------------------
# 3. Simulación de episodios
# ---------------------------------------------------------
n_episodios = 30
for ep in range(n_episodios):
    pos = (0,0)
    while pos not in [(2,2),(2,0)]:
        accion = politica_fija(pos)
        nueva_pos = mover(pos, accion)
        r = recompensa(nueva_pos)
        V[pos] += alpha * (r + gamma * V[nueva_pos] - V[pos])
        pos = nueva_pos

# ---------------------------------------------------------
# 4. Mostrar resultados
# ---------------------------------------------------------
print("==============================================")
print("TRACE APRENDIZAJE POR REFUERZO PASIVO")
print("==============================================\n")

for i in range(grid_size[0]):
    fila = " ".join(f"{V[i,j]:6.2f}" for j in range(grid_size[1]))
    print(f"Fila {i}: {fila}")

# Nota:
# - El agente no decide sus acciones, solo evalúa la política dada.
# - Este método se llama "Aprendizaje Temporal" (TD Learning).
# - Sirve como base para el aprendizaje activo y Q-Learning.
