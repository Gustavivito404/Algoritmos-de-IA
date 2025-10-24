# =========================================================
# 🔹 BÚSQUEDA EN PROFUNDIDAD (DFS) – versión simple
# ---------------------------------------------------------
# Descripción:
#   Implementación básica de Depth-First Search (DFS)
#   en un grafo no ponderado. Versión recursiva simple:
#   - Explora en profundidad sin reconstrucción de camino.
#   - Muestra el orden de visita.
#
# Grafo usado (mismo layout que en BFS/UCS):
#
#        ┌───B───D
#        │   │
#        A   E───G
#        │   │
#        C───F
#
# =========================================================

def dfs_simple(graph, start, visited=None):
    """DFS recursiva simple: recorre el grafo en profundidad."""
    if visited is None:
        visited = set()

    # Marca el nodo actual como visitado
    visited.add(start)
    print(f"Visitando: {start}")

    # Explora cada vecino no visitado
    for neighbor in graph.get(start, []):
        if neighbor not in visited:
            dfs_simple(graph, neighbor, visited)

    return visited


if __name__ == "__main__":
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F', 'G'],
        'F': ['C', 'E'],
        'G': ['E']
    }

    start_node = 'A'
    print("Recorrido DFS simple:")
    dfs_simple(graph, start_node)