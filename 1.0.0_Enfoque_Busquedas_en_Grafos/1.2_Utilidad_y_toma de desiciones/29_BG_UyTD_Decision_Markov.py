# =========================================================
# 29 - PROCESO DE DECISIÓN DE MARKOV (MDP)
# ---------------------------------------------------------
# Descripción:
#   Este script define y ejecuta un MDP simple tipo GridWorld.
#   Se muestra su formalización (S, A, P, R, γ) y se calcula
#   la utilidad óptima usando la ecuación de Bellman.
#
# =========================================================

from typing import Dict, Tuple
import numpy as np

# ---------------------------------------------------------
# 1. Definición formal del MDP
# ---------------------------------------------------------
# Estados (S): todas las celdas válidas
# Acciones (A): ↑, ↓, ←, →
# Transiciones (P): deterministas (100% hacia el destino)
# Recompensas (R): dependen del tipo de celda

grid = [
    ['S', '.', '+'],
    ['.', 'X', '.'],
    ['-', '.', '.']
]

filas = len(grid)
cols = len(grid[0])
acciones = ['↑', '↓', '←', '→']
gamma = 0.95

# ---------------------------------------------------------
# 2. Recompensas R(s)
# ---------------------------------------------------------
R: Dict[Tuple[int, int], float] = {}
for i in range(filas):
    for j in range(cols):
        simbolo = grid[i][j]
        if simbolo == '+':
            R[(i, j)] = 10.0
        elif simbolo == '-':
            R[(i, j)] = -10.0
        elif simbolo == 'X':
            R[(i, j)] = 0.0
        else:
            R[(i, j)] = -1.0  # penalización por movimiento

# ---------------------------------------------------------
# 3. Funciones auxiliares
# ---------------------------------------------------------
def es_valido(i: int, j: int) -> bool:
    return 0 <= i < filas and 0 <= j < cols and grid[i][j] != 'X'

def mover(i: int, j: int, accion: str) -> Tuple[int, int]:
    if accion == '↑': nuevo = (i - 1, j)
    elif accion == '↓': nuevo = (i + 1, j)
    elif accion == '←': nuevo = (i, j - 1)
    elif accion == '→': nuevo = (i, j + 1)
    else: nuevo = (i, j)
    if es_valido(*nuevo): return nuevo
    return (i, j)

# ---------------------------------------------------------
# 4. Ecuación de Bellman para MDP
# ---------------------------------------------------------
def bellman_update(V: np.ndarray, i: int, j: int, gamma: float) -> float:
    """Aplica la actualización de Bellman para un estado (i,j)."""
    valores_accion = []
    for a in acciones:
        (ni, nj) = mover(i, j, a)
        valor = R[(i, j)] + gamma * V[ni, nj]
        valores_accion.append(valor)
    return max(valores_accion)

# ---------------------------------------------------------
# 5. Ejecución del MDP mediante iteración de valores
# ---------------------------------------------------------
def resolver_mdp(max_iter=20, gamma=0.95):
    V = np.zeros((filas, cols))
    politica = np.full((filas, cols), '', dtype='<U1')

    for k in range(max_iter):
        print(f"\n--- Iteración {k+1} ---")
        for i in range(filas):
            for j in range(cols):
                if grid[i][j] == 'X':
                    continue
                V[i, j] = bellman_update(V, i, j, gamma)

        print(V)

    # Derivar política óptima
    for i in range(filas):
        for j in range(cols):
            if grid[i][j] == 'X':
                continue
            mejor_accion = None
            mejor_valor = float('-inf')
            for a in acciones:
                (ni, nj) = mover(i, j, a)
                valor = R[(i, j)] + gamma * V[ni, nj]
                if valor > mejor_valor:
                    mejor_valor = valor
                    mejor_accion = a
            politica[i, j] = mejor_accion

    return V, politica

# ---------------------------------------------------------
# 6. MAIN: Ejecución y resultados
# ---------------------------------------------------------
if __name__ == "__main__":
    V, politica = resolver_mdp(max_iter=20, gamma=0.95)

    print("\n==============================================")
    print("VALORES ÓPTIMOS (V*)")
    print("==============================================")
    for fila in V:
        print(["%6.2f" % v for v in fila])

    print("\nPOLÍTICA ÓPTIMA (π*)")
    print("==============================================")
    for fila in politica:
        print(["  " + a if a else " ---" for a in fila])

    # Nota:
    # - Este código formaliza todo el entorno como un MDP.
    # - Usa la ecuación de Bellman para encontrar V* y π*.
    # - A partir de aquí, puedes conectar este MDP con métodos
    #   de Aprendizaje por Refuerzo (Q-Learning, SARSA, etc.)
