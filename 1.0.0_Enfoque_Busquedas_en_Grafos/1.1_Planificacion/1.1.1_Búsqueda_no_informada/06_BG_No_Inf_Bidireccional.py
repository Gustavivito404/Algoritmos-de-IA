# =========================================================
# BÚSQUEDA BIDIRECCIONAL (Bidirectional BFS)
# ---------------------------------------------------------
# Descripción:
#   Corre dos búsquedas en anchura (BFS) al mismo tiempo:
#   - una desde el nodo inicial (lado INI)
#   - otra desde el nodo meta (lado META)
#
#   Cada lado expande capa por capa.
#   En cuanto un nodo aparece en ambos conjuntos visitados,
#   significa que las dos ondas de búsqueda se encontraron
#   y podemos reconstruir el camino completo.
#
#   Esta técnica reduce mucho el número de nodos explorados
#   cuando el grafo es grande y no ponderado.
#
# Grafo usado:
#
#              ┌───B───D───H
#              │   │    \
#              │   │     I
#              │   │      \
#              A   E───G───J
#              │   │       │
#              C── F──K────L
#                \ │
#                  M───N
#
# Nodos: A, B, C, D, E, F, G, H, I, J, K, L, M, N
#
# =========================================================

from collections import deque
from typing import Dict, List, Any, Optional, Set, Tuple

def reconstruir_camino_bidireccional(
    meet_node: Any,
    parent_start: Dict[Any, Any],
    parent_goal: Dict[Any, Any],
    start: Any,
    goal: Any
) -> List[Any]:
    """
    Reconstruye el camino final usando:
      - parent_start: padres desde el lado inicio
      - parent_goal: padres desde el lado meta
    meet_node es el nodo donde ambas búsquedas se encontraron.
    """

    # Camino desde start hasta meet_node (usando parent_start hacia atrás)
    izquierda = []
    n = meet_node
    while n is not None:
        izquierda.append(n)
        n = parent_start.get(n)
    izquierda.reverse()  # ahora está start -> ... -> meet_node

    # Camino desde meet_node hasta goal (usando parent_goal hacia atrás)
    derecha = []
    n = meet_node
    while n is not None:
        derecha.append(n)
        n = parent_goal.get(n)
    # derecha está meet_node -> ... -> goal
    # Para no repetir el meet_node, quitamos el primero
    derecha = derecha[1:]

    return izquierda + derecha


def bidirectional_bfs_trazado(
    graph: Dict[Any, List[Any]],
    start: Any,
    goal: Any
) -> Optional[List[Any]]:
    """
    Búsqueda bidireccional con trazas.
    - Alterna entre expandir desde el inicio y desde la meta.
    - Muestra:
        Paso,
        Lado que está expandiendo (INI/META),
        Nodo que saca de su cola,
        Estado de ambas colas,
        Visitados de ambos lados.
    """

    # Caso trivial
    if start == goal:
        return [start]

    # Fronteras tipo BFS (colas FIFO)
    cola_start = deque([start])
    cola_goal  = deque([goal])

    # Conjuntos de visitados
    visit_start: Set[Any] = {start}
    visit_goal:  Set[Any] = {goal}

    # Padres desde cada lado (para reconstrucción)
    parent_start = {start: None}
    parent_goal  = {goal: None}

    paso = 0

    print(f"{'Paso':<5} {'Lado':<8} {'Nodo actual':<12} {'Cola_start':<28} {'Cola_goal':<28} {'Visit_start':<22} {'Visit_goal'}")
    print("─" * 150)

    # Mientras haya algo que expandir en ambos lados
    while cola_start and cola_goal:

        # ---------- Expansión desde el lado INI ----------
        if cola_start:
            actual_s = cola_start.popleft()
            paso += 1
            print(f"{paso:<5} {'INI':<8} {str(actual_s):<12} {str(list(cola_start)):<28} {str(list(cola_goal)):<28} {sorted(list(visit_start))!s:<22} {sorted(list(visit_goal))!s}")

            # ¿Nos encontramos con el otro lado?
            if actual_s in visit_goal:
                print("\n✅ Coincidencia detectada (lado inicio)\n")
                return reconstruir_camino_bidireccional(actual_s, parent_start, parent_goal, start, goal)

            # Expandir vecinos desde inicio
            for nb in graph.get(actual_s, []):
                if nb not in visit_start:
                    visit_start.add(nb)
                    parent_start[nb] = actual_s
                    cola_start.append(nb)

        # ---------- Expansión desde el lado META ----------
        if cola_goal:
            actual_g = cola_goal.popleft()
            paso += 1
            print(f"{paso:<5} {'META':<8} {str(actual_g):<12} {str(list(cola_start)):<28} {str(list(cola_goal)):<28} {sorted(list(visit_start))!s:<22} {sorted(list(visit_goal))!s}")

            # ¿Coincidimos aquí?
            if actual_g in visit_start:
                print("\n✅ Coincidencia detectada (lado meta)\n")
                return reconstruir_camino_bidireccional(actual_g, parent_start, parent_goal, start, goal)

            # Expandir vecinos desde meta
            for nb in graph.get(actual_g, []):
                if nb not in visit_goal:
                    visit_goal.add(nb)
                    parent_goal[nb] = actual_g
                    cola_goal.append(nb)

    print("\n❌ No hay conexión entre start y goal")
    return None


if __name__ == "__main__":
    # Grafo grande extendido (A–N)
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F', 'M'],
        'D': ['B', 'H', 'I'],
        'E': ['B', 'F', 'G'],
        'F': ['C', 'E', 'K', 'M'],
        'G': ['E', 'J'],
        'H': ['D'],
        'I': ['D', 'J'],
        'J': ['G', 'I', 'L'],
        'K': ['F', 'L'],
        'L': ['K', 'J'],
        'M': ['C', 'F', 'N'],
        'N': ['M']
    }

    inicio = 'A'
    meta   = 'L'

    camino = bidirectional_bfs_trazado(graph, inicio, meta)

    print(f"\nCamino encontrado de {inicio} a {meta}: {camino}")
