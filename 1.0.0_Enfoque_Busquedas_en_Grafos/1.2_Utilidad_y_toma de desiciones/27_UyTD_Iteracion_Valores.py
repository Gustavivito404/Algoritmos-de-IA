# =========================================================
# 27 - ITERACIÓN DE VALORES (Value Iteration)
# ---------------------------------------------------------
# Descripción:
#   Implementa el algoritmo de iteración de valores
#   para un entorno tipo gridworld 3x3.
#
#   Se calcula el valor óptimo de cada celda y
#   la política óptima (mejor movimiento por celda)
#   aplicando la ecuación de Bellman.
#
# =========================================================

from typing import Dict, Tuple
import numpy as np

# ---------------------------------------------------------
# 1. Definición del entorno
# ---------------------------------------------------------
grid = [
    ['S', '.', '+'],
    ['.', 'X', '.'],
    ['-', '.', '.']
]

filas = len(grid)
cols = len(grid[0])

acciones = ['↑', '↓', '←', '→']
gamma = 0.9   # factor de descuento
recompensa_movimiento = -1.0

# Recompensas específicas
R: Dict[Tuple[int, int], float] = {}
for i in range(filas):
    for j in range(cols):
        simbolo = grid[i][j]
        if simbolo == '+':
            R[(i, j)] = 10.0
        elif simbolo == '-':
            R[(i, j)] = -10.0
        elif simbolo == 'X':
            R[(i, j)] = 0.0  # pared
        else:
            R[(i, j)] = recompensa_movimiento

# ---------------------------------------------------------
# 2. Funciones auxiliares
# ---------------------------------------------------------
def es_valido(i: int, j: int) -> bool:
    """Verifica si una celda es válida (dentro de rango y no pared)."""
    if 0 <= i < filas and 0 <= j < cols and grid[i][j] != 'X':
        return True
    return False

def mover(i: int, j: int, accion: str) -> Tuple[int, int]:
    """Calcula la nueva posición según la acción elegida."""
    if accion == '↑':
        nuevo = (i - 1, j)
    elif accion == '↓':
        nuevo = (i + 1, j)
    elif accion == '←':
        nuevo = (i, j - 1)
    elif accion == '→':
        nuevo = (i, j + 1)
    else:
        nuevo = (i, j)

    if es_valido(*nuevo):
        return nuevo
    return (i, j)  # si choca, se queda donde estaba

# ---------------------------------------------------------
# 3. Iteración de valores
# ---------------------------------------------------------
def value_iteration(max_iter=20, gamma=0.9, theta=1e-4):
    """
    Ejecuta el algoritmo de Iteración de Valores hasta converger.
    """
    V = np.zeros((filas, cols))
    politica = np.full((filas, cols), '', dtype='<U1')

    for k in range(max_iter):
        delta = 0
        print(f"\n--- Iteración {k+1} ---")
        for i in range(filas):
            for j in range(cols):
                if grid[i][j] == 'X':
                    continue  # estados terminales o pared

                valores_accion = []
                for a in acciones:
                    (ni, nj) = mover(i, j, a)
                    valor = R[(i, j)] + gamma * V[ni, nj]
                    valores_accion.append((valor, a))

                mejor_valor, mejor_accion = max(valores_accion, key=lambda x: x[0])
                delta = max(delta, abs(mejor_valor - V[i, j]))
                V[i, j] = mejor_valor
                politica[i, j] = mejor_accion

        print(V)
        if delta < theta:
            break

    return V, politica

# ---------------------------------------------------------
# 4. MAIN: Ejecutar iteración de valores
# ---------------------------------------------------------
if __name__ == "__main__":
    V, politica = value_iteration(max_iter=20, gamma=0.95)

    print("\n==============================================")
    print("VALORES FINALES")
    print("==============================================")
    for fila in V:
        print(["%6.2f" % v for v in fila])

    print("\nPOLÍTICA ÓPTIMA")
    print("==============================================")
    for fila in politica:
        print(["  " + a if a else " ---" for a in fila])

    # Nota:
    # - Este algoritmo itera aplicando la ecuación de Bellman
    #   hasta que los valores de estado convergen.
    #
    # - Cada celda obtiene un valor proporcional a su distancia
    #   a la meta (+10) o a la trampa (-10), considerando el
    #   costo de moverse (-0.1) y el descuento gamma.
