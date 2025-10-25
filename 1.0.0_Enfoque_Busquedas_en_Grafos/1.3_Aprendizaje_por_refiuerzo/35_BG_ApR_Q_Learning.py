# =========================================================
# 35 - Q-LEARNING
# ---------------------------------------------------------
# Descripción:
#   En este módulo implementamos Q-Learning clásico.
#
#   Q-Learning es un algoritmo de Aprendizaje por Refuerzo
#   libre de modelo ("model-free") que aprende directamente
#   la función Q(s,a) ~ valor esperado acumulado de tomar
#   la acción 'a' en el estado 's' y luego seguir la política
#   greedy respecto a Q.
#
#   Actualización central de Q-Learning:
#
#   Q(s,a) ← Q(s,a) + α · [ r + γ · max_a' Q(s',a') − Q(s,a) ]
#
#   Diferencia clave con el Aprendizaje Activo del módulo 34:
#     - En Q-Learning la actualización usa SIEMPRE el
#       "mejor futuro posible" (max_a' Q(s',a')), no la acción
#       que realmente se tomó después.
#
#   Eso lo hace "off-policy": aprende la política óptima
#   aunque la exploración siga otra política (ε-greedy).
#
# Entorno:
#   - Grid 3x3
#   - Recompensa +10 en (2,2) [meta]
#   - Recompensa -10 en (2,0) [trampa]
#   - Recompensa -1 en otros estados
#
# Política de exploración:
#   ε-greedy (a veces explora, a veces explota lo mejor)
#
# Trazas:
#   - Imprimimos episodios
#   - Para cada paso: estado, acción, recompensa, nuevo Q
#
# Al final:
#   - Mostramos la tabla Q(s,a)
#   - Mostramos la política aprendida (mejor acción por celda)
#
# =========================================================

import numpy as np
import random
from typing import Dict, Tuple

# ---------------------------------------------------------
# 1. Definición del entorno (grid 3x3)
# ---------------------------------------------------------
# Estados = posiciones (i,j) con i,j ∈ {0,1,2}
# Acciones = ↑, ↓, ←, →
# Transiciones deterministas, excepto que el borde te deja en el mismo lugar
# Recompensas:
#   (2,2) -> +10   (meta)
#   (2,0) -> -10   (trampa)
#   otro  -> -1    (costo por moverse)
# ---------------------------------------------------------

ACCIONES = ["↑","↓","←","→"]
GRID_SIZE = (3,3)

def mover(pos: Tuple[int,int], accion: str) -> Tuple[int,int]:
    """Dada una posición y una acción, regresa la nueva posición."""
    i,j = pos
    if accion == "↑":
        i = max(i-1,0)
    elif accion == "↓":
        i = min(i+1,GRID_SIZE[0]-1)
    elif accion == "←":
        j = max(j-1,0)
    elif accion == "→":
        j = min(j+1,GRID_SIZE[1]-1)
    return (i,j)

def recompensa(pos: Tuple[int,int]) -> float:
    """Recompensa inmediata al LLEGAR a 'pos'."""
    if pos == (2,2):
        return 10.0
    if pos == (2,0):
        return -10.0
    return -1.0

def es_terminal(pos: Tuple[int,int]) -> bool:
    """Estados terminales: meta o trampa."""
    return pos in [(2,2),(2,0)]

# ---------------------------------------------------------
# 2. Parámetros del aprendizaje
# ---------------------------------------------------------
alpha = 0.2      # tasa de aprendizaje
gamma = 0.9      # descuento futuro
epsilon = 0.2    # exploración ε-greedy
episodios = 30   # cuántas veces dejamos que el agente intente llegar

# Q-Table: diccionario Q[(estado,accion)] = valor
Q: Dict[Tuple[Tuple[int,int], str], float] = {}

# ---------------------------------------------------------
# 3. Funciones auxiliares
# ---------------------------------------------------------
def valor_Q(pos: Tuple[int,int], accion: str) -> float:
    """Lee Q(pos,accion) con default=0."""
    return Q.get((pos,accion), 0.0)

def mejor_accion(pos: Tuple[int,int]) -> str:
    """Regresa la acción con mayor Q en ese estado (rompe empates al azar)."""
    valores = [valor_Q(pos,a) for a in ACCIONES]
    max_v = max(valores)
    mejores = [a for a,v in zip(ACCIONES,valores) if v == max_v]
    return random.choice(mejores)

def elegir_accion_epsilon_greedy(pos: Tuple[int,int], epsilon: float) -> str:
    """Con prob ε explora acción aleatoria, con 1-ε usa greedy."""
    if random.random() < epsilon:
        return random.choice(ACCIONES)
    else:
        return mejor_accion(pos)

def max_Q_estado(pos: Tuple[int,int]) -> float:
    """max_a' Q(pos,a')"""
    return max(valor_Q(pos,a) for a in ACCIONES)

# ---------------------------------------------------------
# 4. Entrenamiento Q-Learning con traza
# ---------------------------------------------------------
print("==============================================")
print("TRACE Q-LEARNING")
print("==============================================\n")

for ep in range(episodios):
    pos = (0,0)  # empezamos siempre arriba-izquierda
    print(f"--- Episodio {ep+1} ---")
    while not es_terminal(pos):
        # 1) Elegir acción con ε-greedy
        a = elegir_accion_epsilon_greedy(pos, epsilon)

        # 2) Ejecutar acción
        nueva_pos = mover(pos, a)
        r = recompensa(nueva_pos)

        # 3) Target de Q-Learning:
        #    r + γ * max_a' Q(s',a')
        objetivo = r + gamma * (0.0 if es_terminal(nueva_pos) else max_Q_estado(nueva_pos))

        # 4) Actualización Q(s,a)
        viejo = valor_Q(pos,a)
        Q[(pos,a)] = viejo + alpha * (objetivo - viejo)

        # 5) Trazas paso a paso
        print(f"Estado {pos} -> Acción '{a}' -> Estado {nueva_pos}")
        print(f"  Recompensa r = {r:.2f}")
        print(f"  Q antes = {viejo:.3f}")
        print(f"  Objetivo TD = {objetivo:.3f}")
        print(f"  Q después = {Q[(pos,a)]:.3f}\n")

        # 6) Mover al siguiente estado
        pos = nueva_pos

    print(f"*** Episodio {ep+1} terminado en {pos} (terminal) ***\n")

# ---------------------------------------------------------
# 5. Construcción de la POLÍTICA GREEDY final
# ---------------------------------------------------------
politica = np.empty(GRID_SIZE, dtype='<U2')
for i in range(GRID_SIZE[0]):
    for j in range(GRID_SIZE[1]):
        s = (i,j)
        if es_terminal(s):
            politica[i,j] = "X"
        else:
            politica[i,j] = mejor_accion(s)

# ---------------------------------------------------------
# 6. Mostrar resultados finales
# ---------------------------------------------------------
print("==============================================")
print("Q(s,a) FINAL (aprox)")
print("==============================================")
for (estado, accion), val in sorted(Q.items()):
    print(f"Q{estado, accion} = {val:.3f}")
print("")

print("POLÍTICA GREEDY APRENDIDA")
print("==============================================")
for i in range(GRID_SIZE[0]):
    fila = " ".join(f"{politica[i,j]:>3s}" for j in range(GRID_SIZE[1]))
    print(f"Fila {i}: {fila}")

# Nota:
# - Q-Learning es off-policy:
#     Aprende usando max_a' Q(s',a') incluso si esa acción
#     no fue realmente tomada después.
#
# - A diferencia del aprendizaje pasivo (33), aquí no seguimos
#   una política fija: la vamos mejorando con la experiencia.
#
# - A diferencia del aprendizaje activo (34), aquí la actualización
#   es explícitamente la regla estándar de Q-Learning, que converge
#   hacia la Q* óptima con suficiente exploración y episodios.
#
# - En el siguiente módulo (36 Exploración vs. Explotación)
#   vamos a analizar el papel de ε y por qué no debe ser ni 0 ni 1.
