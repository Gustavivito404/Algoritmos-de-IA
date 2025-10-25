# =========================================================
# 12 - BÚSQUEDA TABÚ (Tabu Search)
# ---------------------------------------------------------
# Descripción:
#   Búsqueda Tabú es una búsqueda local mejorada.
#   Parte de una solución inicial y se mueve entre vecinos,
#   como Hill Climbing, pero mantiene memoria de las
#   soluciones visitadas recientemente en una lista TABÚ.
#
#   Idea:
#       - Explorar mejoras locales.
#       - Evitar regresar a soluciones recientes
#         (para no ciclarse ni quedar atrapado).
#
#   Esto le permite "escapar" de óptimos locales.
#
# Características:
#   - Estado actual + lista tabú (memoria corta plazo).
#   - Permite movimientos que NO mejoran, si sirven para salir del atasco.
#   - Selecciona el mejor vecino permitido (no tabú).
#
# Definiciones en este script:
#   - Cada nodo del grafo es un "estado".
#   - La función objetivo es MINIMIZAR h(n),
#     donde h(n) es la distancia Manhattan hasta la meta 'L'.
#   - La lista tabú guarda nodos recientemente visitados.
#
# Heurística:
#   h(n) = |x_n - x_L| + |y_n - y_L|
#
# Grafo usado (mismo que en A*, Hill Climbing):
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
# Flujo general de la Búsqueda Tabú:
#
#   actual = inicio
#   mejor_global = actual
#   tabu = [actual]
#
#   repetir hasta MAX_ITER:
#       generar lista de vecinos de 'actual'
#       filtrar los que están en tabu (prohibidos)
#       si TODOS están tabú, relajamos la restricción
#       elegir el vecino con menor h()
#       movernos a ese vecino
#       si mejora el mejor_global (h más baja), actualizarlo
#       actualizar la lista tabu con el nuevo nodo
#       recortar tabu a un tamaño fijo (TABU_TAM)
#
#   resultado final = mejor_global
#
# Nota:
#   Tabú no garantiza llegar EXACTAMENTE a la meta,
#   pero suele acercarse más que Hill Climbing cuando
#   Hill Climbing se estancaría.
#
# Parámetros importantes:
#   - MAX_ITER: cuántos pasos máximo damos.
#   - TABU_TAM: cuántos estados recientes recordamos como prohibidos.
#
# =========================================================

from typing import Dict, List, Tuple
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
# Coordenadas aproximadas de cada nodo en el plano
# (mismas que en heurística Manhattan usada antes)
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
    h(n): Distancia Manhattan estimada desde 'nodo' hasta la meta.
    Queremos MINIMIZAR este valor.
    """
    (x1, y1) = coords[nodo]
    (x2, y2) = coords[meta]
    return fabs(x1 - x2) + fabs(y1 - y2)


def mejor_vecino_no_tabu(actual: str, tabu: List[str]) -> Tuple[str, float]:
    """
    Selecciona el mejor vecino del nodo 'actual' que NO esté en la lista tabú.
    Criterio: menor h(n).
    Si todos los vecinos están tabú, entonces ignoramos tabu y
    elegimos el mejor (esto es una relajación típica en Tabu Search).
    Regresa (vecino_elegido, h_elegida).
    """
    vecinos = graph[actual]

    # Calculamos (nodo_vecino, h(vecino))
    evaluaciones = []
    for v in vecinos:
        h_v = heuristica_manhattan(v, META)
        evaluaciones.append((v, h_v))

    # Intento 1: filtrar por los que NO son tabú
    candidatos = [(v, h_v) for (v, h_v) in evaluaciones if v not in tabu]

    # Si todos están en tabú, relajo la prohibición (elegiré igual)
    if len(candidatos) == 0:
        candidatos = evaluaciones  # sin filtro tabú

    # Ordenar por mejor (menor h)
    candidatos.sort(key=lambda par: par[1])
    return candidatos[0]  # (mejor_vecino, su_h)


def tabu_search(inicio: str,
                meta: str,
                MAX_ITER: int = 15,
                TABU_TAM: int = 4) -> Tuple[List[str], str, float]:
    """
    Búsqueda Tabú básica:
    - inicio: nodo inicial
    - meta: nodo objetivo deseado
    - MAX_ITER: pasos máximos a intentar
    - TABU_TAM: tamaño máximo de la lista tabú
    Devuelve:
      - historial_recorrido: la secuencia real de nodos visitados
      - mejor_nodo_global: el nodo "más prometedor" encontrado
      - h(mejor_nodo_global): qué tan cerca quedó de la meta
    """
    actual = inicio

    # Lista tabú (memoria reciente)
    tabu: List[str] = [actual]

    # Mejor solución global encontrada hasta ahora
    mejor_nodo_global = actual
    mejor_h_global = heuristica_manhattan(actual, meta)

    historial = [actual]

    print("==============================================")
    print("TRACE Tabu Search")
    print("==============================================\n")
    print(f"Inicio en {actual} con h={mejor_h_global:.1f}")
    print(f"Meta deseada: {meta}\n")

    for paso in range(MAX_ITER):

        # ¿Ya llegamos exactamente a la meta?
        if actual == meta:
            print(f">> META alcanzada exactamente ({meta}) en iteración {paso}")
            break

        # Elegir el mejor vecino que no esté tabú
        vecino_elegido, h_elegida = mejor_vecino_no_tabu(actual, tabu)

        print(f"[Iter {paso}] Nodo actual: {actual}")
        print(f"  h({actual}) = {heuristica_manhattan(actual, meta):.1f}")
        print(f"  Tabú actual: {tabu}")
        print(f"  -> Mejor vecino permitido: {vecino_elegido}  h={h_elegida:.1f}")

        # Movernos (aunque NO mejore)
        actual = vecino_elegido
        historial.append(actual)

        # Actualizar mejor global si mejora
        if h_elegida < mejor_h_global:
            mejor_h_global = h_elegida
            mejor_nodo_global = actual
            print(f"  * Nuevo mejor global: {mejor_nodo_global} con h={mejor_h_global:.1f}")

        # Actualizar lista tabú con el nodo al que acabamos de llegar
        tabu.append(actual)
        # Limitar el tamaño de la lista tabú
        if len(tabu) > TABU_TAM:
            eliminado = tabu.pop(0)
            print(f"  (Tabú overflow) Removemos {eliminado} de tabú")

        print("")

    print("==============================================")
    print("RESULTADO TABÚ")
    print("==============================================")
    print("Historial recorrido :", " -> ".join(historial))
    print("Mejor nodo global   :", mejor_nodo_global)
    print("h(mejor nodo global):", mejor_h_global)
    print("¿Llegamos a la meta?:", "Sí" if historial[-1] == meta else "No")
    print("")

    return historial, mejor_nodo_global, mejor_h_global


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    # Ejecutar la búsqueda Tabú desde 'A' tratando de llegar hacia 'L'
    trayectoria, mejor_nodo, mejor_h = tabu_search(
        inicio='A',
        meta='L',
        MAX_ITER=15,
        TABU_TAM=4
    )

    print("Resumen final:")
    print("Trayectoria completa:", " -> ".join(trayectoria))
    print("Mejor nodo alcanzado globalmente:", mejor_nodo)
    print("Heurística (distancia estimada a meta) en mejor nodo:", mejor_h)
    print("Nota: Búsqueda Tabú puede salir del estancamiento porque")
    print("      permite moverse incluso si NO mejora, y evita ciclos con la lista tabú.\n")
