# =========================================================
# 14 - BÚSQUEDA DE HAZ LOCAL (Local Beam Search)
# ---------------------------------------------------------
# Descripción:
#   La Búsqueda de Haz Local (Local Beam Search) es una mejora
#   sobre Hill Climbing. En lugar de tener solo UNA posición
#   actual, mantiene K candidatos al mismo tiempo.
#
#   En cada iteración:
#       - Desde TODOS los candidatos actuales, generamos
#         TODOS sus vecinos.
#       - Elegimos los K mejores (según heurística h).
#       - Esos K mejores se vuelven la "población" de la
#         siguiente iteración.
#
#   Así evitamos que toda la búsqueda se quede atorada
#   en un solo óptimo local, porque varios intentos
#   están explorando en paralelo.
#
#   Importante:
#       • Si cualquiera de los K candidatos alcanza la meta,
#         podemos parar.
#
# Parámetros principales:
#   - K (ancho del haz): cuántos estados mantenemos vivos.
#   - MAX_ITER: cuántas rondas de expansión hacemos.
#
# Heurística:
#   Usamos la misma h(n) que en los otros algoritmos:
#       h(n) = distancia Manhattan hasta la meta 'L'
#   y la meta es MINIMIZAR h(n).
#
# Grafo y meta:
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
# Coordenadas usadas (ya actualizadas para espaciar nodos):
#
#    'A': (0, 2),
#    'B': (1, 0),
#    'D': (2, 0),
#    'H': (4, 0),
#
#    'I': (3, 1),
#    'E': (1, 2),
#    'G': (2, 2),
#    'J': (4, 2),
#
#    'C': (0, 3),
#    'F': (1, 3),
#    'K': (2, 3),
#    'L': (4, 3),
#
#    'M': (1, 4),
#    'N': (2, 4)
#
# Flujo resumido del algoritmo:
#
#   población = [ k estados iniciales distintos ]
#
#   repetir:
#       - si alguien en población es la meta -> parar
#       - generar todos los vecinos de toda la población
#       - evaluar h() de cada vecino
#       - quedarnos con los K mejores únicos
#       - eso es la nueva población
#
#   regresamos:
#       • el historial de poblaciones por iteración
#       • el mejor nodo visto
#
# Nota:
#   Esta versión educativa NO hace "Beam Search Estocástico",
#   es la versión determinista básica.
#
# =========================================================

from typing import Dict, List, Tuple, Set
from math import fabs

# ---------------------------------------------------------
# Grafo (no dirigido)
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
# Coordenadas (tu versión corregida con más espacio)
# ---------------------------------------------------------
coords: Dict[str, Tuple[int, int]] = {
    'A': (0, 2),
    'B': (1, 0),
    'D': (2, 0),
    'H': (4, 0),

    'I': (3, 1),
    'E': (1, 2),
    'G': (2, 2),
    'J': (4, 2),

    'C': (0, 3),
    'F': (1, 3),
    'K': (2, 3),
    'L': (4, 3),

    'M': (1, 4),
    'N': (2, 4)
}

META = 'L'


def heuristica_manhattan(nodo: str, meta: str = META) -> float:
    """
    h(n) = |x_n - x_meta| + |y_n - y_meta|
    """
    (x1, y1) = coords[nodo]
    (x2, y2) = coords[meta]
    return fabs(x1 - x2) + fabs(y1 - y2)


def expandir_poblacion(poblacion: List[str]) -> List[str]:
    """
    Dada la lista de estados actuales (población),
    regresa TODOS los vecinos posibles (con repetidos).
    """
    vecinos_totales: List[str] = []
    for nodo in poblacion:
        vecinos_totales.extend(graph[nodo])
    return vecinos_totales


def k_mejores_candidatos(candidatos: List[str], K: int) -> List[str]:
    """
    Dada una lista de nodos candidatos (puede tener repetidos),
    ordena por h(n) ascendente y regresa los K mejores **únicos**.
    """
    # Quitamos duplicados pero conservamos la mejor h encontrada para cada uno
    mejor_h_por_nodo: Dict[str, float] = {}
    for nodo in candidatos:
        h_val = heuristica_manhattan(nodo, META)
        if (nodo not in mejor_h_por_nodo) or (h_val < mejor_h_por_nodo[nodo]):
            mejor_h_por_nodo[nodo] = h_val

    # Ordenar nodos únicos por h(n) creciente
    ordenados = sorted(mejor_h_por_nodo.items(), key=lambda par: par[1])

    # Tomar solo los primeros K nodos
    topK = [n for (n, _) in ordenados[:K]]
    return topK


def local_beam_search(
    iniciales: List[str],
    meta: str,
    K: int,
    MAX_ITER: int = 10
) -> Tuple[List[List[str]], str, float]:
    """
    Búsqueda de Haz Local (determinista):
    - iniciales: lista inicial de nodos (tamaño K inicial típico)
    - meta: nodo objetivo
    - K: ancho del haz
    - MAX_ITER: iteraciones máximas

    Regresa:
      - historial_poblaciones: lista con la población en cada iteración
      - mejor_nodo_global: el mejor estado visto (menor h)
      - mejor_h_global: h de ese mejor estado
    """

    poblacion_actual: List[str] = iniciales[:K]  # asegurarnos tamaño K
    historial_poblaciones: List[List[str]] = [poblacion_actual[:]]

    # Mejor global encontrado
    mejor_nodo = min(poblacion_actual, key=lambda n: heuristica_manhattan(n, meta))
    mejor_h = heuristica_manhattan(mejor_nodo, meta)

    print("==============================================")
    print("TRACE Local Beam Search")
    print("==============================================\n")

    print(f"Población inicial (K={K}): {poblacion_actual}")
    for n in poblacion_actual:
        print(f"  h({n}) = {heuristica_manhattan(n, meta):.1f}")
    print("")

    for iteracion in range(MAX_ITER):

        # ¿Alguno alcanzó la meta?
        if meta in poblacion_actual:
            print(f">> META '{meta}' encontrada en la población en la iteración {iteracion}")
            break

        # Expandimos TODOS los nodos de la población
        candidatos = expandir_poblacion(poblacion_actual)

        print(f"[Iter {iteracion}] Población actual: {poblacion_actual}")
        print("  Vecinos generados (candidatos):", candidatos)
        print("  h() de los candidatos únicos:")

        # Escogemos los K mejores candidatos únicos por h
        nueva_poblacion = k_mejores_candidatos(candidatos, K)

        # Mostrar evaluaciones de la nueva población
        for n in nueva_poblacion:
            print(f"    {n}: h={heuristica_manhattan(n, meta):.1f}")

        # Actualizamos el mejor global si encontramos algo mejor
        candidato_mejor_local = min(nueva_poblacion, key=lambda n: heuristica_manhattan(n, meta))
        h_candidato_mejor = heuristica_manhattan(candidato_mejor_local, meta)
        if h_candidato_mejor < mejor_h:
            mejor_h = h_candidato_mejor
            mejor_nodo = candidato_mejor_local
            print(f"  * Nuevo mejor global: {mejor_nodo} con h={mejor_h:.1f}")

        # Actualizamos población
        poblacion_actual = nueva_poblacion
        historial_poblaciones.append(poblacion_actual[:])

        print("")

    print("==============================================")
    print("RESULTADO LOCAL BEAM SEARCH")
    print("==============================================")
    print("Historial de poblaciones:")
    for i, pob in enumerate(historial_poblaciones):
        etiquetas = [f"{n}(h={heuristica_manhattan(n, meta):.1f})" for n in pob]
        print(f"  Iter {i}: {etiquetas}")

    print("Mejor nodo global   :", mejor_nodo)
    print("h(mejor nodo global):", mejor_h)
    print("¿Meta alcanzada en última población?:", "Sí" if meta in poblacion_actual else "No")
    print("")

    return historial_poblaciones, mejor_nodo, mejor_h


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    # Ejemplo:
    # Vamos a iniciar con K=3 nodos distintos.
    # Elegimos algunos nodos razonablemente separados:
    #   'A' (izquierda),
    #   'B' (arriba),
    #   'C' (parte baja izquierda).
    K = 3
    poblacion_inicial = ['A', 'B', 'C']

    historial, mejor_nodo, mejor_h = local_beam_search(
        iniciales=poblacion_inicial,
        meta='L',
        K=K,
        MAX_ITER=10
    )

    print("Resumen final:")
    print("Historial poblaciones:", historial)
    print("Mejor nodo global:", mejor_nodo)
    print("h(mejor nodo global):", mejor_h)
