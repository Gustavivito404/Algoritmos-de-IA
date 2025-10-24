# =========================================================
# 08 - HEURÍSTICAS (versión extendida con estimación geométrica)
# ---------------------------------------------------------
# Descripción:
#   En este script construimos y comparamos dos heurísticas
#   hacia la meta 'L' en el mismo grafo de referencia:
#
#   1) Una heurística MANUAL (h_manual):
#      Asignada por "intuición humana", basada en qué tan
#      lejos parece cada nodo de la meta.
#
#   2) Una heurística GEOMÉTRICA (h_geo):
#      Le damos coordenadas (x,y) aproximadas a cada nodo
#      según el diagrama visual del grafo, y calculamos
#      la distancia tipo Manhattan hasta 'L'.
#
#      h_geo(n) = |x_n - x_L| + |y_n - y_L|
#
#      Esto imita cómo se construyen heurísticas reales:
#      rápidas de calcular y razonables, sin tener que
#      explorar todo el grafo.
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

from typing import Dict, Tuple
from math import fabs

# ---------------------------------------------------------
# 1. Grafo (no dirigido)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2. Heurística manual (h_manual)
# ---------------------------------------------------------
# Esta fue asignada por "intuición" sobre cuán lejos
# parece cada nodo de la meta 'L'. No es exacta, pero
# ordena los nodos por cercanía aproximada.
#
# Nota importante:
#   Ajustamos valores con tu observación sobre H:
#   H puede llegar a L en 4 saltos (H-D-I-J-L),
#   así que le damos un valor en la misma categoría
#   que nodos que están a distancia parecida.
# ---------------------------------------------------------
h_manual: Dict[str, int] = {
    'L': 0,   # meta
    'J': 1, 'K': 1,
    'G': 2, 'I': 2, 'F': 2,
    'E': 3, 'C': 3, 'B': 3, 'M': 3,
    'A': 4, 'D': 4, 'N': 4, 'H': 4
}


def heuristica_manual(nodo: str) -> int:
    """Heurística manual hacia 'L'."""
    return h_manual.get(nodo, 9999)


# ---------------------------------------------------------
# 3. Coordenadas aproximadas para cada nodo
# ---------------------------------------------------------
# Vamos a darle a cada nodo una posición (x, y) que más
# o menos respete el diagrama visual:
#
#   E está más o menos "arriba-centro",
#   J está en la derecha,
#   L está abajo-derecha,
#   M y N cuelgan abajo a la izquierda,
#   A está más a la izquierda del bloque,
#   etc.
#
# Estas coordenadas NO son "reales", son un modelo mental
# del grafo en 2D para poder calcular una heurística rápida.
#
# Mientras la ubicación conserve la idea de que
# L está en (3,2) y los nodos "más cerca" están cerca
# en el plano, h_geo va a tener sentido.
# ---------------------------------------------------------

coords: Dict[str, Tuple[int, int]] = {
    # columna X , fila Y
    'A': (0, 1),
    'B': (1, 0),
    'D': (2, 0),
    'H': (3, 0),

    'I': (3, 1),
    'E': (1, 1),
    'G': (2, 1),
    'J': (3, 1),

    'C': (0, 2),
    'F': (1, 2),
    'K': (2, 2),
    'L': (3, 2),

    'M': (1, 3),
    'N': (2, 3)
}

# Notas:
# - (más a la derecha) => más cerca de L en x
# - (más abajo)        => más cerca de L en y si L está abajo-derecha
# - A está a la izquierda
# - L está en (3,2)


# ---------------------------------------------------------
# 4. Heurística geométrica (h_geo)
# ---------------------------------------------------------
# Vamos a usar Distancia Manhattan a 'L':
#
#   h_geo(n) = |x_n - x_L| + |y_n - y_L|
#
# Esta es una aproximación típica en robótica y pathfinding
# en mapas tipo grid, porque estima “cuántos pasos en grilla”
# faltan si pudieras moverte horizontal/verticalmente.
#
# IMPORTANTE:
#   Esta heurística NO explora el grafo completo.
#   Solo usa una fórmula con coordenadas.
#   Por eso es "barata" y escalable.
# ---------------------------------------------------------

META = 'L'

def heuristica_geometrica(nodo: str) -> float:
    """Heurística tipo Manhattan hasta L usando coords."""
    (x1, y1) = coords[nodo]
    (x2, y2) = coords[META]
    return fabs(x1 - x2) + fabs(y1 - y2)


# ---------------------------------------------------------
# 5. Comparación / salida
# ---------------------------------------------------------
# Vamos a imprimir una tabla con:
#   - Nodo
#   - h_manual(n)
#   - h_geo(n)
#   - Nota sobre cuál "parece" más optimista
#
# Interpretación:
#   - h_manual viene de intuición humana sobre el grafo.
#   - h_geo viene de una fórmula geométrica estable.
#
# En problemas reales:
#   • h_geo sería algo como distancia Euclidiana / Manhattan.
#   • h_manual sería experiencia humana, reglas, etc.
#
# Ambas son heurísticas válidas si:
#   - Son baratas de calcular
#   - Distinguen nodos
#   - Apuntan hacia la meta
# ---------------------------------------------------------

if __name__ == "__main__":
    print(f"{'Nodo':<5} {'h_manual':<10} {'h_geo':<10} Comentario")
    print("─" * 60)

    # Ordenamos por lo que la heurística geométrica cree más prometedor
    for nodo in sorted(graph.keys(), key=lambda n: heuristica_geometrica(n)):
        hm = heuristica_manual(nodo)
        hg = heuristica_geometrica(nodo)

        # comentario simple para ayudarte a leer
        if nodo == META:
            nota = "Meta"
        elif hg <= 1:
            nota = "Está pegado a L geométricamente"
        elif hg <= 2:
            nota = "Muy cerca en el plano"
        elif hg <= 3:
            nota = "Distancia media"
        else:
            nota = "Lejos en el plano"

        print(f"{nodo:<5} {hm:<10} {hg:<10.1f} {nota}")
