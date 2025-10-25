# =========================================================
# 10 - BÚSQUEDAS A* y AO*
# ---------------------------------------------------------
# Descripción:
#   En este script implementamos dos algoritmos de búsqueda
#   informada que utilizan heurísticas para guiarse hacia
#   la meta de manera eficiente:
#
#   1) A* (A estrella):
#      Combina el costo real g(n) con la estimación heurística h(n)
#      para expandir el nodo con el menor costo total:
#          f(n) = g(n) + h(n)
#      Garantiza la ruta óptima si h(n) es admisible.
#
#   2) AO* (And-Or Search):
#      Extiende el principio de A* para grafos AND/OR,
#      donde algunos nodos requieren cumplir varias metas
#      simultáneamente (AND) y otros son alternativas (OR).
#      Se usa en planificación y resolución de tareas compuestas.
#
# Grafo usado en A*:
#
#              ┌───B───D───H
#              │   │    \
#              │   │     I
#              │   │      \
#              A   E───G───J
#              │   │       │
#              C── F──K────L   <-- Meta
#                \ │
#                  M───N
#
# Nodos: A, B, C, D, E, F, G, H, I, J, K, L, M, N
#
# Características:
#   - A*: búsqueda óptima con frontera ordenada por f = g + h.
#   - AO*: búsqueda en grafos con nodos compuestos (AND y OR).
#
# =========================================================

from typing import Dict, List, Tuple, Optional
from math import fabs
import heapq
from functools import lru_cache

# ---------------------------------------------------------
# A* SEARCH
# ---------------------------------------------------------
graph: Dict[str, List[str]] = {
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

coords: Dict[str, Tuple[int, int]] = {
    'A': (0, 1), 'B': (1, 0), 'D': (2, 0), 'H': (3, 0),
    'I': (3, 1), 'E': (1, 1), 'G': (2, 1), 'J': (3, 1),
    'C': (0, 2), 'F': (1, 2), 'K': (2, 2), 'L': (3, 2),
    'M': (1, 3), 'N': (2, 3)
}

META = 'L'

def heuristica_manhattan(nodo: str, meta: str = META) -> float:
    (x1, y1) = coords[nodo]
    (x2, y2) = coords[meta]
    return fabs(x1 - x2) + fabs(y1 - y2)

def reconstruir_camino(padres: Dict[str, Optional[str]], actual: str) -> List[str]:
    ruta = [actual]
    while padres[actual] is not None:
        actual = padres[actual]
        ruta.append(actual)
    ruta.reverse()
    return ruta

def a_estrella(inicio: str, meta: str) -> Tuple[List[str], Dict[str, float]]:
    g: Dict[str, float] = {inicio: 0.0}
    padres: Dict[str, Optional[str]] = {inicio: None}
    frontera: List[Tuple[float, str]] = []
    heapq.heappush(frontera, (heuristica_manhattan(inicio, meta), inicio))
    cerrados = set()
    paso = 0

    print("==============================================")
    print("TRACE A* Search")
    print("==============================================\n")

    while frontera:
        f_actual, actual = heapq.heappop(frontera)

        print(f"[Paso {paso}] {actual}  f={f_actual:.1f}  g={g[actual]:.1f}  h={heuristica_manhattan(actual, meta):.1f}")
        if actual == meta:
            print("\n>> META encontrada:", actual)
            ruta_final = reconstruir_camino(padres, actual)
            print(">> Ruta final:", " -> ".join(ruta_final))
            print(">> Costo total:", g[actual])
            return ruta_final, g

        cerrados.add(actual)

        for vecino in graph[actual]:
            costo_vecino = g[actual] + 1
            if vecino in g and costo_vecino >= g[vecino]:
                continue
            if vecino in cerrados:
                continue
            g[vecino] = costo_vecino
            padres[vecino] = actual
            f_vecino = g[vecino] + heuristica_manhattan(vecino, meta)
            heapq.heappush(frontera, (f_vecino, vecino))
            print(f"   + {vecino}: g={g[vecino]:.1f}, h={heuristica_manhattan(vecino, meta):.1f}, f={f_vecino:.1f}")

        paso += 1
        print("")

    print("No hay ruta.")
    return [], g


# ---------------------------------------------------------
# AO* SEARCH
# ---------------------------------------------------------
grafo_ao: Dict[str, List[Tuple[str, List[str], float]]] = {
    'Start': [
        ('AND', ['Plan1_A', 'Plan1_B'], 5.0),
        ('OR',  ['Plan2'],              7.0)
    ],
    'Plan1_A': [('OR', ['Goal'], 2.0)],
    'Plan1_B': [('OR', ['Goal'], 4.0)],
    'Plan2':   [('OR', ['Goal'], 3.0)],
    'Goal': []
}

@lru_cache(maxsize=None)
def costo_minimo_ao(nodo: str) -> float:
    if nodo == 'Goal':
        return 0.0
    opciones = grafo_ao[nodo]
    costos = []
    for tipo, hijos, base in opciones:
        if tipo == 'AND':
            subtotal = base + sum(costo_minimo_ao(h) for h in hijos)
        else:
            subtotal = base + min(costo_minimo_ao(h) for h in hijos)
        costos.append(subtotal)
    return min(costos)

def explicar_ao(nodo: str, indent: int = 0):
    pref = "  " * indent
    if nodo == 'Goal':
        print(pref + f"- {nodo}: costo 0")
        return
    print(pref + f"- {nodo}:")
    for tipo, hijos, base in grafo_ao[nodo]:
        if tipo == 'AND':
            total = base + sum(costo_minimo_ao(h) for h in hijos)
            print(pref + f"   AND base={base} total={total}")
            for h in hijos: explicar_ao(h, indent + 2)
        else:
            best = min(hijos, key=lambda hh: costo_minimo_ao(hh))
            total = base + costo_minimo_ao(best)
            print(pref + f"   OR base={base} total={total}, mejor={best}")
            explicar_ao(best, indent + 2)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n================ A* ================\n")
    ruta, costos = a_estrella('A', 'L')
    print("\nRuta final:", " -> ".join(ruta))
    print("Costo total:", costos.get('L'))

    print("\n================ AO* ================\n")
    explicar_ao('Start')
    print("\nCosto total Start:", costo_minimo_ao('Start'))
