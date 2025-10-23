# =========================================================
# BÚSQUEDA DE COSTO UNIFORME (UCS)
# ---------------------------------------------------------
# Descripción:
#   Implementación didáctica de Uniform Cost Search (Dijkstra sin heurística)
#   sobre un grafo con pesos no negativos. Muestra el proceso paso a paso:
#   nodo extraído, cola de prioridad y mejor costo conocido.
#
# Diferencia clave vs BFS:
#   - BFS usa COLA FIFO y minimiza número de saltos (todas las aristas = 1).
#   - UCS usa COLA DE PRIORIDAD y minimiza el COSTO ACUMULADO.
#
# Grafo (mismo layout que usaste; ahora con pesos):
#
#        ┌───B───D
#        │ 1 │ 2
#        A   E───G
#        │ 4 │1
#        C───F
#          1  3
#
# Pesos por arista (simétricos):
#   A–B:1, A–C:4, B–D:2, B–E:5, C–F:1, E–G:1, E–F:3
# =========================================================

import heapq
from typing import Dict, Any, Optional, List, Tuple

def ucs_trazado(graph: Dict[Any, Dict[Any, float]], start: Any, goal: Any) -> Optional[List[Any]]:
    """
    Uniform Cost Search (UCS) con trazas:
      - graph: diccionario {nodo: {vecino: costo, ...}}
      - start, goal: nodos de inicio y meta
    Retorna:
      - lista con el camino de costo mínimo, o None si no existe.
    """
    # Cola de prioridad: (costo_acumulado, nodo)
    pq: List[Tuple[float, Any]] = [(0.0, start)]
    # Mejor costo conocido para cada nodo
    best_cost = {start: 0.0}
    # Padres para reconstrucción de camino
    parent = {start: None}

    paso = 0
    print(f"{'Paso':<5} {'Nodo actual':<12} {'Costo':<8} {'Cola prioridad (costo,nodo)':<40} {'MejorCosto'}")
    print("─" * 110)

    while pq:
        costo_act, actual = heapq.heappop(pq)
        paso += 1

        # Si este pop no es óptimo (hay un costo mejor conocido), se descarta
        if costo_act > best_cost.get(actual, float('inf')):
            # Traza igualmente (útil para ver descartes)
            print(f"{paso:<5} {str(actual):<12} {costo_act:<8.2f} {str(pq):<40} {sorted(best_cost.items())}")
            continue

        # Traza del estado
        print(f"{paso:<5} {str(actual):<12} {costo_act:<8.2f} {str(pq):<40} {sorted(best_cost.items())}")

        if actual == goal:
            # Reconstrucción del camino óptimo
            ruta = []
            nodo = actual
            while nodo is not None:
                ruta.append(nodo)
                nodo = parent[nodo]
            ruta.reverse()
            print("\n✅ Meta encontrada (costo mínimo)\n")
            return ruta

        # Relajar aristas
        for vecino, w in graph.get(actual, {}).items():
            nuevo_costo = costo_act + w
            if nuevo_costo < best_cost.get(vecino, float('inf')):
                best_cost[vecino] = nuevo_costo
                parent[vecino] = actual
                heapq.heappush(pq, (nuevo_costo, vecino))

    print("\n❌ No se encontró la meta")
    return None


if __name__ == "__main__":
    # Grafo con pesos (no negativos), simétricos
    grafo_pesos = {
        'A': {'B': 1, 'C': 4},
        'B': {'A': 1, 'D': 2, 'E': 5},
        'C': {'A': 4, 'F': 1},
        'D': {'B': 2},
        'E': {'B': 5, 'G': 1, 'F': 3},
        'F': {'C': 1, 'E': 3},
        'G': {'E': 1}
    }

    inicio, meta = 'A', 'G'
    camino = ucs_trazado(grafo_pesos, inicio, meta)
    print(f"Camino óptimo por costo: {camino}")
