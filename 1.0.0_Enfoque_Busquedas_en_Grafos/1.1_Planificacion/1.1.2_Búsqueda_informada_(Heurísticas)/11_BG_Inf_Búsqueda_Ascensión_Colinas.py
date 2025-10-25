# =========================================================
# 11 - BÚSQUEDA DE ASCENSIÓN DE COLINAS (Hill Climbing)
# ---------------------------------------------------------
# Descripción:
#   Hill Climbing (ascensión de colinas) es un algoritmo
#   de búsqueda local que intenta mejorar paso a paso
#   moviéndose SIEMPRE al vecino que parece "mejor"
#   según una función de evaluación.
#
#   Idea mental:
#       Estoy en una colina en la niebla.
#       Solo veo alrededor de mí (mis vecinos).
#       Me muevo hacia donde sube más rápido.
#       Me detengo cuando ya no puedo subir más.
#
# Características:
#   - No guarda una frontera global como A*.
#   - No recuerda por dónde ha pasado.
#   - Solo conoce su estado actual y sus vecinos.
#   - Puede atascarse en óptimos locales.
#
# Matemática / Criterio:
#   Queremos MINIMIZAR h(n) = distancia estimada a la meta.
#
#   En cada paso:
#       escogemos el vecino con h más baja.
#       si ese vecino NO mejora (h no baja), paramos.
#
#   Es puramente "codicioso local".
#
# Grafo usado (mismo grafo que A*, heurística Manhattan a 'L'):
#
#              ┌───B───D───H
#              │   │    \
#              │   │     I
#              │   │      \
#              A   E───G───J
#              │   │       │
#              C── F──K────L   <-- L = meta
#                \ │
#                  M───N
#
# Nodos: A, B, C, D, E, F, G, H, I, J, K, L, M, N
#
# Flujo de ejecución (versión simple):
#
#   estado_actual = inicio
#   mientras True:
#       revisar vecinos del estado_actual
#       elegir el vecino con h más baja
#       si h(vecino_mejor) < h(estado_actual):
#           movernos a ese vecino
#       si no mejora:
#           detener (estancado)
#
# =========================================================

from typing import Dict, List, Tuple
from math import fabs

# ---------------------------------------------------------
# Grafo no dirigido
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

# ---------------------------------------------------------
# Coordenadas aproximadas (mismas que en heurística / A*)
# ---------------------------------------------------------
coords: Dict[str, Tuple[int, int]] = {
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

META = 'L'


def heuristica_manhattan(nodo: str, meta: str = META) -> float:
    """
    h(n): Distancia Manhattan desde 'nodo' hasta la META.
    En Hill Climbing vamos a INTENTAR MINIMIZAR esta h.
    """
    (x1, y1) = coords[nodo]
    (x2, y2) = coords[meta]
    return fabs(x1 - x2) + fabs(y1 - y2)


def hill_climbing(inicio: str, meta: str) -> List[str]:
    """
    Búsqueda de Ascensión de Colinas (versión básica "steepest-ascent").
    - En cada paso elegimos el vecino con la mejor heurística (menor h).
    - Si ningún vecino mejora, nos detenemos.
    - Regresamos el camino que tomamos paso a paso.
    """

    actual = inicio
    camino = [actual]  # guardamos el recorrido real que hace el algoritmo
    paso = 0

    print("==============================================")
    print("TRACE Hill Climbing")
    print("==============================================\n")
    print(f"Inicio en nodo {actual} con h={heuristica_manhattan(actual, meta):.1f}\n")

    while True:
        if actual == meta:
            print(f">> META alcanzada: {actual}")
            break

        vecinos = graph[actual]

        # Calculamos la heurística de cada vecino
        evaluaciones = []
        for v in vecinos:
            h_v = heuristica_manhattan(v, meta)
            evaluaciones.append((h_v, v))

        # Ordenamos por h más baja (mejor candidato primero)
        evaluaciones.sort(key=lambda t: t[0])

        print(f"[Paso {paso}] Nodo actual: {actual}")
        print(f"  h({actual}) = {heuristica_manhattan(actual, meta):.1f}")
        print(f"  Vecinos y sus h():")
        for (hval, v) in evaluaciones:
            print(f"    {v}: h={hval:.1f}")

        # Tomamos el mejor vecino según h
        mejor_h, mejor_vecino = evaluaciones[0]
        print(f"  Mejor vecino candidato: {mejor_vecino} con h={mejor_h:.1f}")

        # ¿Mejora real? (tiene que bajar h)
        if mejor_h < heuristica_manhattan(actual, meta):
            print(f"  Movimiento: {actual} -> {mejor_vecino} (mejora h)\n")
            actual = mejor_vecino
            camino.append(actual)
            paso += 1
        else:
            print("  No hay mejora. Algoritmo se detiene (óptimo local / meseta / estancado).\n")
            break

    print("Camino recorrido por Hill Climbing:", " -> ".join(camino))
    print("h(final) =", heuristica_manhattan(actual, meta))
    print("")

    return camino


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    # Ejemplo de ejecución:
    # Probamos iniciar en 'A' e intentar llegar hacia 'L'
    ruta = hill_climbing('A', 'L')

    print("Resumen final:")
    print("Recorrido hecho por Hill Climbing:", " -> ".join(ruta))
    print("Nodo final alcanzado:", ruta[-1])
    print("¿Llegó a la meta?:", "Sí" if ruta[-1] == 'L' else "No")
