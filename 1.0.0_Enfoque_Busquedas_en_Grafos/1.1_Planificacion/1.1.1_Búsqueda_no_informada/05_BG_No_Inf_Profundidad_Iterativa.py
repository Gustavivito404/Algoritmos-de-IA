# =========================================================
# BÚSQUEDA EN PROFUNDIDAD ITERATIVA (IDS)
# ---------------------------------------------------------
# Descripción:
#   Iterative Deepening Search = Búsqueda en profundidad
#   con límite creciente. Intenta primero L=0, luego L=1,
#   luego L=2, etc., hasta encontrar la meta.
#
# Ventaja:
#   - Usa muy poca memoria (como DFS).
#   - Encuentra caminos mínimos en número de pasos (como BFS),
#     en grafos no ponderados.
#
# Relación:
#   - IDS llama internamente a DLS (Depth-Limited Search).
#
# Grafo usado:
#
#        ┌───B───D
#        │   │
#        A   E───G
#        │   │
#        C───F
#
# =========================================================

from typing import Dict, List, Tuple, Optional, Set

Result = Tuple[str, Optional[List[str]]]  # ('success'|'cutoff'|'failure', camino|None)

def depth_limited_search(graph: Dict[str, List[str]], start: str, goal: str, L: int) -> Result:
    """
    Subrutina DLS (idéntica en comportamiento a la que ya tenemos):
    Hace DFS recursivo pero sin pasar el límite L.
    """
    visited: Set[str] = set()
    parent = {start: None}

    def rec(u: str, depth: int) -> str:
        if u not in visited:
            visited.add(u)

        # ¿Meta encontrada?
        if u == goal:
            return 'success'

        # ¿Alcanzamos el límite?
        if depth == L:
            return 'cutoff'

        cutoff_occurred = False

        for v in graph.get(u, []):
            if v not in visited:
                parent[v] = u
                result = rec(v, depth + 1)

                if result == 'success':
                    return 'success'
                elif result == 'cutoff':
                    cutoff_occurred = True
                # if 'failure': seguimos buscando otros vecinos

        return 'cutoff' if cutoff_occurred else 'failure'

    status = rec(start, 0)

    if status == 'success':
        # reconstruir camino usando 'parent'
        path = []
        node = goal
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return ('success', path)

    if status == 'cutoff':
        return ('cutoff', None)

    return ('failure', None)


def iterative_deepening_search(graph: Dict[str, List[str]], start: str, goal: str, L_max: int) -> Result:
    """
    IDS: va aumentando el límite L = 0,1,2,...,L_max
    y en cada nivel llama depth_limited_search.
    Se detiene tan pronto como encuentra 'success'.
    """
    for L in range(L_max + 1):
        print(f"\n=== Intentando con límite L = {L} ===")
        status, path = depth_limited_search(graph, start, goal, L)

        if status == 'success':
            print("✔ Solución encontrada dentro de este límite.")
            return ('success', path)
        elif status == 'failure':
            print("✘ No hay solución ni más profundo en este subárbol.")
            return ('failure', None)
        else:
            print("↳ Corte por límite (cutoff), aumentando L...")

    return ('failure', None)


if __name__ == "__main__":
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F', 'G'],
        'F': ['C', 'E'],
        'G': ['E', 'H'],
        'H': ['G']
    }
    
    # Puedes cambiar los nodos de inicio y meta aquí:
    # start, goal = 'A', 'G'
    start, goal = 'A', 'H'

    # L_max es el límite máximo que vamos a intentar.
    # En la práctica puedes poner algo grande, pero aquí usamos 4 por ejemplo.
    status, path = iterative_deepening_search(graph, start, goal, L_max=4)

    print("\nEstado final:", status)
    print("Camino encontrado:", path)
