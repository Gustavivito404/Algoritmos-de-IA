# =========================================================
# 37 - BÚSQUEDA DE LA POLÍTICA
# ---------------------------------------------------------
# Descripción:
#   En este módulo hacemos "búsqueda de la política"
#   (policy search / policy iteration estilo tabular),
#   pero visto desde el punto de vista de Aprendizaje por Refuerzo.
#
#   La idea es:
#     1) Suponemos una política π (qué acción tomar en cada estado).
#     2) Evaluamos qué tan buena es (estimamos V(s) bajo π).
#     3) Mejoramos π escogiendo, en cada estado, la acción
#        que maximiza la recompensa esperada usando V(s').
#     4) Repetimos hasta que la política deje de cambiar.
#
#   Esto es básicamente "Iteración de Políticas", pero
#   aquí lo presentamos explícitamente como:
#       "buscar una mejor política"
#   dentro del mismo entorno grid que usamos en los módulos
#   anteriores de MDP y RL.
#
# Entorno:
#   - Grid 3x3
#   - Recompensa +10 en (2,2) [meta]
#   - Recompensa -10 en (2,0) [trampa]
#   - Recompensa -1 en las demás celdas
#   - Transiciones deterministas
#
# Resultado:
#   - Imprimimos las iteraciones de mejora de política
#   - Mostramos la política final óptima
#
# =========================================================

import numpy as np
from typing import Tuple, Dict

# ---------------------------------------------------------
# 1. Definimos el entorno
# ---------------------------------------------------------
ACCIONES = ["↑","↓","←","→"]
GRID_SIZE = (3,3)
gamma = 0.95   # descuento futuro
recompensa_mov = -1.0

def es_terminal(pos: Tuple[int,int]) -> bool:
    return pos in [(2,2),(2,0)]

def recompensa(pos: Tuple[int,int]) -> float:
    if pos == (2,2): return 10.0
    if pos == (2,0): return -10.0
    return recompensa_mov

def mover(pos: Tuple[int,int], accion: str) -> Tuple[int,int]:
    (i,j) = pos
    if accion == "↑":
        i = max(i-1,0)
    elif accion == "↓":
        i = min(i+1, GRID_SIZE[0]-1)
    elif accion == "←":
        j = max(j-1,0)
    elif accion == "→":
        j = min(j+1, GRID_SIZE[1]-1)
    return (i,j)

# ---------------------------------------------------------
# 2. Evaluación de política
# ---------------------------------------------------------
def evaluar_politica(politica, theta=1e-4):
    """
    Dada una política π[s] = acción,
    resuelve V(s) aproximadamente aplicando evaluación iterativa:
        V(s) = R(s) + γ * V(s')
    donde s' es el estado siguiente según π[s].
    (Suponemos dinámica determinista.)
    """
    V = np.zeros(GRID_SIZE)

    while True:
        delta = 0.0
        for i in range(GRID_SIZE[0]):
            for j in range(GRID_SIZE[1]):
                s = (i,j)
                if es_terminal(s):
                    # Estados terminales: su valor es su recompensa inmediata.
                    nuevo_val = recompensa(s)
                else:
                    a = politica[i,j]
                    s_next = mover(s, a)
                    r = recompensa(s_next)
                    nuevo_val = r + gamma * V[s_next]

                delta = max(delta, abs(V[i,j] - nuevo_val))
                V[i,j] = nuevo_val

        if delta < theta:
            break

    return V

# ---------------------------------------------------------
# 3. Mejora de política
# ---------------------------------------------------------
def mejorar_politica(V):
    """
    A partir de V(s), elegimos en cada estado la acción
    que maximiza:
        Q(s,a) = R(s→a) + γ * V(s')
    Regresa:
        nueva_política, es_estable(bool)
    """
    nueva_pi = np.full(GRID_SIZE, '', dtype='<U2')
    estable = True

    for i in range(GRID_SIZE[0]):
        for j in range(GRID_SIZE[1]):
            s = (i,j)

            if es_terminal(s):
                nueva_pi[i,j] = "X"
                continue

            mejores_acciones = []
            mejores_valores = []
            best_val = None

            for a in ACCIONES:
                s_next = mover(s,a)
                r = recompensa(s_next)
                q_val = r + gamma * V[s_next]

                if (best_val is None) or (q_val > best_val):
                    best_val = q_val
                    mejores_acciones = [a]
                    mejores_valores = [q_val]
                elif q_val == best_val:
                    mejores_acciones.append(a)
                    mejores_valores.append(q_val)

            # rompo empates con la primera
            nueva_pi[i,j] = mejores_acciones[0]

    return nueva_pi, estable

# ---------------------------------------------------------
# 4. Búsqueda iterativa de la política
# ---------------------------------------------------------
def busqueda_de_politica(max_iter=10):
    """
    - Inicializamos una política arbitraria (por ejemplo, siempre "→").
    - Repetimos:
        1) Evaluar π
        2) Mejorar π usando V
    - Imprimimos trazas en cada iteración.
    """

    politica = np.full(GRID_SIZE, '→', dtype='<U2')
    politica[2,2] = "X"
    politica[2,0] = "X"

    for it in range(max_iter):
        print(f"\n--- Iteración de Política {it+1} ---")

        # 1) Evaluación de la política actual
        V = evaluar_politica(politica)

        print("Valores V(s) estimados bajo la política actual:")
        for fila in V:
            print(["%7.2f" % v for v in fila])
        print("Política actual:")
        for i in range(GRID_SIZE[0]):
            fila_txt = " ".join(f"{politica[i,j]:>3s}" for j in range(GRID_SIZE[1]))
            print(" ", fila_txt)

        # 2) Mejora de la política
        nueva_politica, _ = mejorar_politica(V)

        print("\nPolítica mejorada (greedy respecto a V):")
        for i in range(GRID_SIZE[0]):
            fila_txt = " ".join(f"{nueva_politica[i,j]:>3s}" for j in range(GRID_SIZE[1]))
            print(" ", fila_txt)

        # Si ya no cambia, paramos
        if np.array_equal(nueva_politica, politica):
            print("\n>> La política se volvió estable. Terminamos.")
            return V, politica

        politica = nueva_politica

    return V, politica

# ---------------------------------------------------------
# 5. MAIN: ejecutar búsqueda de la política
# ---------------------------------------------------------
if __name__ == "__main__":
    print("==============================================")
    print("TRACE BÚSQUEDA DE LA POLÍTICA")
    print("==============================================\n")

    V_final, politica_final = busqueda_de_politica(max_iter=10)

    print("\n==============================================")
    print("RESULTADO FINAL BÚSQUEDA DE LA POLÍTICA")
    print("==============================================")
    print("Valores V(s) finales:")
    for fila in V_final:
        print(["%7.2f" % v for v in fila])

    print("\nPolítica final:")
    for i in range(GRID_SIZE[0]):
        fila_txt = " ".join(f"{politica_final[i,j]:>3s}" for j in range(GRID_SIZE[1]))
        print(" ", fila_txt)

    # Nota:
    # - Este script hace exactamente lo que hemos venido haciendo
    #   de forma "manual" con Iteración de Políticas en MDP,
    #   pero ahora lo estamos leyendo en el lente de RL:
    #   "buscar una política buena" en vez de "resolver Bellman".
    #
    # - La política resultante suele dirigir:
    #     • lejos de la trampa (2,0)
    #     • hacia la meta (2,2)
    #   igual que las políticas óptimas que viste en MDP y Q-Learning.
