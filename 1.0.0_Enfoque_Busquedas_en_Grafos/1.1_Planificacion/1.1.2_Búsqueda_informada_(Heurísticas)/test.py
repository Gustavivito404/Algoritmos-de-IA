# =========================================================
# 09 - BÚSQUEDA VORAZ "PRIMERO EL MEJOR"
#        (Greedy Best-First Search)
# ---------------------------------------------------------
# Descripción:
#   Algoritmo de búsqueda INFORMADA.
#
#   Idea central:
#     Siempre expande el nodo que tenga la heurística h(n)
#     más baja, es decir, el que "parece" más cercano
#     a la meta según la estimación.
#
#   Características:
#     - Usa SOLO h(n). No le importa cuánto costó llegar.
#     - Tiende a avanzar directo hacia la meta.
#     - PERO no garantiza que el camino encontrado sea
#       el más corto u óptimo.
#
#   En este ejemplo:
#     - Meta: 'F'
#     - Heurística h(n): distancia Manhattan entre el nodo
#       y la meta 'F', usando coordenadas (x,y) que
#       representan la posición visual aproximada del grafo.
#
# Grafo usado (esquema lógico):
#
#       B ─ C
#     /     |
#   A ─ D   F   (meta)
#       |   |
#       E   I
#       |   |                   
#       G ─ H
#
#   Desde A puedes intentar subir (A→B→C→F)
#   o bajar y rodear (A→D→E→G→H→I→F).
#   Greedy elegirá en cada paso el nodo con h(n) más bajo.
#
# =========================================================

from typing import Dict, List, Tuple, Optional
import heapq

# ---------------------------------------------------------
# Grafo (no dirigido)
# ---------------------------------------------------------
graph: Dict[str, List[str]] = {
    'A': ['D', 'B'],
    'B': ['A', 'C'],
    'C': ['B', 'F'],
    'D': ['A', 'E'],
    'E': ['D', 'G'],
    'F': ['C', 'I'],   # F es la meta
    'G': ['E', 'H'],
    'H': ['G', 'I'],
    'I': ['H', 'F']
}

# ---------------------------------------------------------
# Coordenadas (x,y) aproximadas de cada nodo
# Estas posiciones reflejan el diagrama visual.
#
# Importante:
#   F está en (2,1), y la heurística mide la "distancia"
#   hasta F en el plano.
# ---------------------------------------------------------
coords: Dict[str, Tuple[int, int]] = {
    'A': (0, 1),
    'B': (1, 0),
    'C': (2, 0),
    'F': (2, 1),  # Meta
    'D': (1, 1),
    'I': (2, 2),
    'E': (1, 2),
    'H': (2, 3),
    'G': (1, 3)
}

META = 'F'

# ---------------------------------------------------------
# Heurística h(n): Distancia Manhattan hasta F
#
#   h(n) = |x_n - x_F| + |y_n - y_F|
#
# Esta h(n) NO es el costo real: es una estimación de
# qué tan cerca "parece" estar el nodo de la meta.
# ---------------------------------------------------------
def h(nodo: str) -> float:
    (x1, y1) = coords[nodo]
    (x2, y2) = coords[META]
    return abs(x1 - x2) + abs(y1 - y2)

# ---------------------------------------------------------
# Algoritmo Greedy Best-First Search
# ---------------------------------------------------------
def greedy_best_first_search(
    graph: Dict[str, List[str]],
    start: str,
    goal: str
) -> Optional[List[str]]:
    """
    Implementación tipo GRAPH-SEARCH:
      - Usa una cola de prioridad ordenada por h(n).
      - Lleva 'visitados' para no ciclar.
      - Usa 'parent' para reconstruir el camino.

    Imprime paso a paso:
      Paso | Nodo actual | h(n) | Frontera | Visitados
    """

    # Cola de prioridad (min-heap) con tuplas (h(nodo), nodo)
    frontera: List[Tuple[float, str]] = []
    heapq.heappush(frontera, (h(start), start))

    visitados = set()
    parent: Dict[str, Optional[str]] = {start: None}

    paso = 0
    print("====================================================")
    print(f"Greedy Best-First Search: inicio '{start}' → meta '{goal}'")
    print("====================================================")
    print(f"{'Paso':<5} {'Nodo actual':<12} {'h(n)':<6} {'Frontera (h,node)':<35} {'Visitados'}")
    print("─" * 110)

    while frontera:
        # Sacar el nodo más prometedor según h(n)
        heur_actual, actual = heapq.heappop(frontera)
        paso += 1

        print(f"{paso:<5} {actual:<12} {heur_actual:<6.1f} {str(frontera):<35} {sorted(list(visitados))}")

        # Evitar reprocesar nodos ya visitados
        if actual in visitados:
            continue

        visitados.add(actual)

        # ¿Meta alcanzada?
        if actual == goal:
            # reconstrucción del camino usando 'parent'
            camino = []
            nodo = actual
            while nodo is not None:
                camino.append(nodo)
                nodo = parent[nodo]
            camino.reverse()
            print("\n✅ Meta alcanzada (Greedy)\n")
            return camino

        # Expandimos vecinos
        for vecino in graph.get(actual, []):
            if vecino not in visitados:
                heapq.heappush(frontera, (h(vecino), vecino))
                # Registrar el padre sólo la primera vez
                if vecino not in parent:
                    parent[vecino] = actual

    print("\n❌ No se encontró la meta\n")
    return None

# ---------------------------------------------------------
# Ejecución del ejemplo principal:
#   Inicio = 'A'
#   Meta   = 'F'
# ---------------------------------------------------------
if __name__ == "__main__":
    inicio = 'A'
    meta = 'F'

    camino = greedy_best_first_search(graph, inicio, meta)
    print(f"Camino sugerido por Greedy de {inicio} a {meta}: {camino}")
