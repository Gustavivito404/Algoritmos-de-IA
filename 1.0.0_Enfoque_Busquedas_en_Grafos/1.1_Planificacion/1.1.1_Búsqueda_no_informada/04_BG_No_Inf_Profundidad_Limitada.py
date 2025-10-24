# =========================================================
# BÚSQUEDA EN PROFUNDIDAD LIMITADA (DLS)
# ---------------------------------------------------------
# Descripción:
#   DFS con límite de profundidad L. Detiene la recursión cuando
#   depth == L. Reporta:
#     - 'success' con camino si llega a la meta
#     - 'cutoff' si recorta (hay más por explorar pero el límite lo impide)
#     - 'failure' si no hay solución en el subárbol sin recortes
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

def dls(graph: Dict[str, List[str]], start: str, goal: str, L: int) -> Result:
    """
    Depth-Limited Search (DLS) – recursivo.
    """
    visited: Set[str] = set()
    parent = {start: None}

    def rec(u: str, depth: int) -> str:
        """
        Retorna 'success' | 'cutoff' | 'failure' y actualiza parent si éxito.
        """
        nonlocal visited
        visited.add(u)

        # ¿Meta?
        if u == goal:
            return 'success'

        # ¿Tope de profundidad?
        if depth == L:
            return 'cutoff'

        cutoff_occurred = False
        for v in graph.get(u, []):
            if v not in visited:
                parent[v] = u
                result = rec(v, depth + 1)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result == 'success':
                    return 'success'
                # si 'failure', seguimos con otros vecinos

        return 'cutoff' if cutoff_occurred else 'failure'

    status = rec(start, 0)

    if status == 'success':
        # reconstruir camino
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

    start, goal, L = 'A', 'H', 4 # Cambiar L para probar distintos límites
    status, path = dls(graph, start, goal, L)
    print(f"Estado: {status}")
    print(f"Camino: {path}")
