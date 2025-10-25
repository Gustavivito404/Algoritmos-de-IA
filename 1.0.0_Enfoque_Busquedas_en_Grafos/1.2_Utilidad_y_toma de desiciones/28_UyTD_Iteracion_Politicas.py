# =========================================================
# 28 - ITERACIÓN DE POLÍTICAS (Policy Iteration)
# ---------------------------------------------------------
# Descripción:
#   Resuelve el mismo gridworld que usamos en Value Iteration,
#   pero alternando entre evaluación y mejora de política.
#
#   Al converger, π(s) = política óptima y V(s) = valores óptimos.
#
# =========================================================

from typing import Dict, Tuple
import numpy as np

# ---------------------------------------------------------
# 1. Definición del entorno (mismo del anterior)
# ---------------------------------------------------------
grid = [
    ['S', '.', '+'],
    ['.', 'X', '.'],
    ['-', '.', '.']
]

filas = len(grid)
cols = len(grid[0])
acciones = ['↑', '↓', '←', '→']

gamma = 0.95
recompensa_movimiento = -1.0

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
            R[(i, j)] = recompensa_movimiento

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
# 2. Evaluación de política
# ---------------------------------------------------------
def evaluar_politica(politica, V, gamma=0.95, theta=1e-4):
    while True:
        delta = 0
        for i in range(filas):
            for j in range(cols):
                if grid[i][j] == 'X':
                    continue
                accion = politica[i][j]
                (ni, nj) = mover(i, j, accion)
                nuevo_valor = R[(i, j)] + gamma * V[ni, nj]
                delta = max(delta, abs(V[i, j] - nuevo_valor))
                V[i, j] = nuevo_valor
        if delta < theta:
            break
    return V

# ---------------------------------------------------------
# 3. Mejora de política
# ---------------------------------------------------------
def mejorar_politica(V, politica, gamma=0.95):
    estable = True
    for i in range(filas):
        for j in range(cols):
            if grid[i][j] == 'X':
                continue
            valores_accion = []
            for a in acciones:
                (ni, nj) = mover(i, j, a)
                valores_accion.append((R[(i, j)] + gamma * V[ni, nj], a))
            mejor_accion = max(valores_accion, key=lambda x: x[0])[1]
            if mejor_accion != politica[i][j]:
                estable = False
                politica[i][j] = mejor_accion
    return politica, estable

# ---------------------------------------------------------
# 4. Iteración principal de políticas
# ---------------------------------------------------------
def policy_iteration():
    # Inicializar política y valores
    politica = np.full((filas, cols), '↑', dtype='<U1')
    V = np.zeros((filas, cols))

    iteracion = 0
    while True:
        iteracion += 1
        print(f"\n--- Iteración {iteracion} ---")
        V = evaluar_politica(politica, V, gamma)
        politica, estable = mejorar_politica(V, politica, gamma)

        print("Valores V:")
        for fila in V:
            print(["%6.2f" % v for v in fila])

        print("Política:")
        for fila in politica:
            print(["  " + a for a in fila])

        if estable:
            break

    return V, politica

# ---------------------------------------------------------
# 5. MAIN: Ejecutar iteración de políticas
# ---------------------------------------------------------
if __name__ == "__main__":
    V, politica = policy_iteration()

    print("\n==============================================")
    print("RESULTADO FINAL - ITERACIÓN DE POLÍTICAS")
    print("==============================================")
    for fila in V:
        print(["%6.2f" % v for v in fila])

    print("\nPOLÍTICA ÓPTIMA")
    print("==============================================")
    for fila in politica:
        print(["  " + a for a in fila])

    # Nota:
    # - Este método converge en menos iteraciones que Value Iteration
    #   porque evalúa y mejora la política en bloques.
    # - Si la política no cambia en una iteración, ya es óptima.
