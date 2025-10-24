# =========================================================
# BÚSQUEDA EN GRAFOS (GRAPH-SEARCH con BFS)
# ---------------------------------------------------------
# Descripción:
#   Implementación de la búsqueda en anchura (BFS)
#   en su versión de "búsqueda en grafo".
#
#   A diferencia de la búsqueda en árbol (TREE-SEARCH),
#   aquí se mantiene un conjunto global de nodos visitados,
#   evitando expandir dos veces el mismo nodo y previniendo
#   ciclos infinitos.
#
#   Retorna el camino más corto (en cantidad de saltos)
#   entre el nodo inicial y la meta.
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
# =========================================================

from collections import deque
from typing import Dict, List, Any, Optional, Set, Tuple

def bfs_graph_search_trazado(
    graph: Dict[Any, List[Any]],
    start: Any,
    goal: Any
) -> Tuple[Optional[List[Any]], List[Any]]:
    """
    Búsqueda en anchura (BFS) en modo GRAPH-SEARCH.
    - Usa un conjunto global de 'visitados' para evitar ciclos.
    - Muestra trazas paso a paso:
        * nodo actual
        * contenido de la cola
        * conjunto de visitados

    Retorna:
        (camino, orden_expansion)
        camino: lista con el recorrido desde start hasta goal
        orden_expansion: lista con el orden en que se expandieron los nodos
    """

    if start not in graph or goal not in graph:
        return None, []

    cola = deque([start])        # Frontera FIFO
    visitados: Set[Any] = {start}
    parent = {start: None}
    orden_expansion: List[Any] = []
    paso = 0

    print(f"{'Paso':<5} {'Nodo actual':<12} {'Cola (FIFO)':<40} {'Visitados'}")
    print("─" * 120)

    while cola:
        actual = cola.popleft()
        paso += 1
        orden_expansion.append(actual)

        print(f"{paso:<5} {str(actual):<12} {str(list(cola)):<40} {sorted(list(visitados))}")

        # ¿Llegamos a la meta?
        if actual == goal:
            # reconstruir camino
            ruta = []
            nodo = actual
            while nodo is not None:
                ruta.append(nodo)
                nodo = parent[nodo]
            ruta.reverse()
            print("\n✅ Meta encontrada\n")
            return ruta, orden_expansion

        # Expandir vecinos
        for vecino in graph.get(actual, []):
            if vecino not in visitados:
                visitados.add(vecino)
                parent[vecino] = actual
                cola.append(vecino)

    print("\n❌ No se encontró la meta\n")
    return None, orden_expansion


if __name__ == "__main__":
    # Grafo oficial de Gus
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

    camino, expansion = bfs_graph_search_trazado(graph, inicio, meta)

    print(f"Camino encontrado A→{meta}: {camino}")
    print(f"Orden de expansión: {expansion}")
