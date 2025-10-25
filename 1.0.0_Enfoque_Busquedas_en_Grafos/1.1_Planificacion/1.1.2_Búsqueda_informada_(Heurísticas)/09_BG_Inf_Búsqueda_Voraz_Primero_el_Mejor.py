# =========================================================
# 09 - BÚSQUEDA VORAZ "PRIMERO EL MEJOR"
#        (Greedy Best-First Search)
# ---------------------------------------------------------
# Descripción:
#   Este algoritmo de búsqueda informada usa una heurística h(n)
#   para decidir qué nodo expandir a continuación.
#
#   Regla principal:
#     Siempre expande el nodo con el valor heurístico h(n)
#     más bajo (el que "parece" más cerca de la meta).
#
#   Importante:
#     - SOLO usa h(n). No toma en cuenta el costo recorrido.
#     - Es rápida y suele ir directo hacia la meta.
#     - PERO no garantiza el camino óptimo.
#
#   En este ejemplo:
#     • La meta es 'L'.
#     • h(n) se define como la distancia Manhattan aproximada
#       entre cada nodo y 'L', usando coordenadas en 2D que
#       representan la posición visual del grafo.
#
# Grafo usado:
#
#              ┌───B──D────H
#              │   │    \
#              │   │     I
#              │   │      \
#              A   E──G──O──J
#              │   │       │
#              C── F──K────L
#                \ │
#                  M───N
#
# =========================================================

from typing import Dict, List, Tuple, Any, Optional
import heapq

# ---------------------------------------------------------
# Grafo oficial (no dirigido)
# ---------------------------------------------------------
graph: Dict[str, List[str]] = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F', 'M'],
    'D': ['B', 'H', 'I'],
    'H': ['D'],
    'I': ['D', 'J', 'O'],   # I conecta a D, a J (directo) y a O
    'E': ['B', 'F', 'G'],
    'F': ['C', 'E', 'K', 'M'],
    'G': ['E', 'O'],
    'O': ['G', 'I', 'J'],   # nodo puente nuevo
    'J': ['O', 'I', 'L'],
    'K': ['F', 'L'],
    'L': ['K', 'J'],
    'M': ['C', 'F', 'N'],
    'N': ['M']
}

# ---------------------------------------------------------
# Coordenadas oficiales (x, y) de cada nodo
# Estas coordenadas reflejan tu layout visual.
# ---------------------------------------------------------
coords: Dict[str, Tuple[int, int]] = {
    'A': (0, 2),
    'B': (1, 0),
    'D': (2, 0),
    'H': (4, 0),
    'I': (3, 1),
    'E': (1, 2),
    'G': (2, 2),
    'O': (3, 2),
    'J': (4, 2),
    'C': (0, 3),
    'F': (1, 3),
    'K': (2, 3),
    'L': (4, 3),
    'M': (1, 4),
    'N': (2, 4)
}

# ---------------------------------------------------------
# Heurística h(n) = Distancia Manhattan hasta 'L'
#
#   h(n) = |x_n - x_L| + |y_n - y_L|
#
# Esta h(n) intenta estimar qué tan "cerca" está un nodo
# de la meta 'L', sin calcular el camino real.
#
# Nota:
#   Esto es una heurística informada, NO un costo real.
#   Greedy va a usar estos valores para priorizar nodos.
# ---------------------------------------------------------
META = 'L'

def h(nodo: str) -> float:
    (x1, y1) = coords[nodo]
    (x2, y2) = coords[META]
    return abs(x1 - x2) + abs(y1 - y2)


# ---------------------------------------------------------
# Búsqueda Voraz Primero el Mejor (Greedy Best-First Search)
# ---------------------------------------------------------
def greedy_best_first_search(
    graph: Dict[str, List[str]],
    start: str,
    goal: str
) -> Optional[List[str]]:
    """
    Implementación tipo GRAPH-SEARCH:
      - Usa una cola de prioridad ordenada por h(n).
      - Mantiene 'visitados' para evitar ciclos.
      - Usa 'parent' para reconstruir el camino al final.

    Esta versión imprime trazas paso a paso:
      Paso | Nodo actual | h(n) | Frontera | Visitados
    """

    # Cola de prioridad con tuplas (h(nodo), nodo)
    frontera: List[Tuple[float, str]] = []
    heapq.heappush(frontera, (h(start), start))

    visitados = set()
    parent: Dict[str, Optional[str]] = {start: None}

    paso = 0
    print("\n====================================================")
    print(f"Greedy Best-First Search: inicio '{start}' → meta '{goal}'")
    print("====================================================")
    print(f"{'Paso':<5} {'Nodo actual':<12} {'h(n)':<6} {'Frontera (h,node)':<40} {'Visitados'}")
    print("─" * 120)

    while frontera:
        # Sacar el nodo más prometedor según h(n)
        heur_actual, actual = heapq.heappop(frontera)
        paso += 1

        print(f"{paso:<5} {actual:<12} {heur_actual:<6.1f} {str(frontera):<40} {sorted(list(visitados))}")

        # Puede pasar que el mismo nodo entró varias veces a la frontera
        # con el mismo padre. Si ya lo cerramos, lo saltamos.
        if actual in visitados:
            continue

        visitados.add(actual)

        # ¿Llegamos a la meta?
        if actual == goal:
            # reconstruir el camino
            camino = []
            nodo = actual
            while nodo is not None:
                camino.append(nodo)
                nodo = parent[nodo]
            camino.reverse()
            print("\n✅ Meta alcanzada (Greedy)\n")
            return camino

        # Expandir vecinos
        for vecino in graph.get(actual, []):
            if vecino not in visitados:
                # A Greedy no le importa g(n), solo h(n)
                heapq.heappush(frontera, (h(vecino), vecino))
                # Guardamos el padre la primera vez que lo vemos
                if vecino not in parent:
                    parent[vecino] = actual

    print("\n❌ No se encontró la meta\n")
    return None


# ---------------------------------------------------------
# Ejecución de pruebas
# ---------------------------------------------------------
if __name__ == "__main__":
    # Vamos a probar varios puntos de inicio
    pruebas_inicio = ['A', 'H', 'M']
    meta = 'L'

    for inicio in pruebas_inicio:
        camino = greedy_best_first_search(graph, inicio, meta)
        print(f"Camino sugerido por Greedy desde {inicio} hasta {meta}: {camino}")
        print("----------------------------------------------------")
