# =========================================================
# BÚSQUEDA EN ANCHURA (BFS)
# 01_BG_No_Inf_Anchura.py
# ---------------------------------------------------------
# Descripción:
#   Este script implementa la búsqueda en anchura (Breadth-First Search)
#   sobre un grafo no ponderado. Es un ejemplo fundamental para entender
#   los algoritmos de búsqueda no informada.
#
# Características:
#   - Usa una cola (FIFO) para recorrer el grafo por niveles.
#   - Encuentra el camino más corto en número de pasos.
#   - Muestra el estado de la cola y los nodos visitados en cada iteración.
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

from collections import deque
from typing import Dict, List, Any, Optional

def bfs_trazado(graph: Dict[Any, List[Any]], start: Any, goal: Any) -> Optional[List[Any]]:
    cola = deque([start])
    visitados = set([start])
    padre = {start: None}
    paso = 0

    print(f"{'Paso':<5} {'Nodo actual':<12} {'Cola (FIFO)':<25} {'Visitados'}")
    print("-" * 70)

    while cola:
        actual = cola.popleft()
        paso += 1

        # Mostrar estado actual
        print(f"{paso:<5} {actual:<12} {str(list(cola)):<25} {sorted(list(visitados))}")


        if actual == goal:
            print("\nMeta encontrada ✅\n")
            ruta = []
            nodo = actual
            while nodo is not None:
                ruta.append(nodo)
                nodo = padre[nodo]
            ruta.reverse()
            return ruta

        # Expandir vecinos no visitados
        for vecino in graph.get(actual, []):
            if vecino not in visitados:
                visitados.add(vecino)
                padre[vecino] = actual
                cola.append(vecino)

    print("\nMeta no encontrada ❌")
    return None

if __name__ == "__main__":
    grafo = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F', 'G'],
        'F': ['C', 'E'],
        'G': ['E']
    }

    inicio, meta = 'A', 'G'
    camino = bfs_trazado(grafo, inicio, meta)
    print("Camino encontrado:", camino)
