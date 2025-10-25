# =========================================================
# 16 - BÚSQUEDA ONLINE
# ---------------------------------------------------------
# Descripción:
#   La búsqueda online es un tipo de búsqueda donde el agente
#   NO conoce el grafo completo desde el inicio.
#
#   En búsquedas "offline" (como A*, Greedy, etc.):
#       - ya tenemos el mapa entero (todas las conexiones).
#
#   En búsqueda "online":
#       - el agente está físicamente en un nodo,
#       - solo conoce sus vecinos inmediatos,
#       - se mueve paso a paso,
#       - va construyendo el mapa mientras explora.
#
#   Esto se parece más a un robot que recorre un laberinto
#   sin tener el plano completo.
#
#   Aquí implementamos una estrategia tipo "Greedy Online":
#     1. Empiezo en un nodo inicial.
#     2. Pregunto: ¿a dónde puedo ir desde aquí?
#     3. Elijo moverme al vecino que parece más prometedor
#        según una heurística h(n).
#     4. Repito hasta que llegue a la meta o me quede atorado.
#
#   Diferencia importante:
#     - El agente solo va recordando lo que ha visto.
#     - No planea toda la ruta completa al inicio.
#
# Heurística usada:
#   h(n) = |x_n - x_L| + |y_n - y_L|
#   (Distancia Manhattan hasta la meta 'L')
#
# Grafo usado (misma topología de todos los ejercicios):
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
# Coordenadas usadas (versión espaciada):
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
# Estrategia que vamos a simular:
#
#   - El agente guarda:
#        • dónde está actualmente
#        • qué nodos ha visitado
#        • qué vecinos ha descubierto de cada nodo
#
#   - En cada paso:
#        1. Observa vecinos del nodo actual (esto sería su "sensor").
#        2. Los añade a un mapa_conocido.
#        3. Decide a qué nodo moverse:
#              elige el vecino con menor h(n), pero
#              prefiere NO repetir nodos ya visitados si hay opción.
#
#   - Se detiene si:
#        • llega a la meta
#        • ya no hay vecinos nuevos útiles
#
# OJO:
#   Esto NO garantiza el mejor camino global.
#   Esto modela más una navegación miope / reactiva con memoria parcial.
#
# =========================================================

from typing import Dict, List, Tuple, Set
from math import fabs

# ---------------------------------------------------------
# Grafo real del mundo (el agente NO lo conoce todo al inicio,
# pero nosotros sí lo usamos "backstage" para simular descubrimiento)
# ---------------------------------------------------------
grafo_real: Dict[str, List[str]] = {
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
# Coordenadas conocidas / heurística Manhattan
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


def h(nodo: str, meta: str = META) -> float:
    """
    h(n): Distancia Manhattan a la meta, usada como guía voraz.
    """
    (x1, y1) = coords[nodo]
    (x2, y2) = coords[meta]
    return fabs(x1 - x2) + fabs(y1 - y2)


def elegir_siguiente(actual: str,
                     vecinos_visibles: List[str],
                     visitados: Set[str]) -> str:
    """
    Política golosa (greedy online):
    - Preferimos vecinos NO visitados todavía.
    - Entre esos, elegimos el que tenga h más baja.
    - Si todos ya fueron visitados, elegimos el de menor h aunque esté repetido.
    """
    # Separamos vecinos nuevos y viejos
    nuevos = [v for v in vecinos_visibles if v not in visitados]
    if nuevos:
        candidatos = nuevos
    else:
        candidatos = vecinos_visibles  # ya no hay novedad, repetir si es necesario

    # Elegimos el de mejor heurística
    mejor = min(candidatos, key=lambda v: h(v))
    return mejor


def busqueda_online(inicio: str,
                    meta: str,
                    MAX_PASOS: int = 20) -> List[str]:
    """
    Simulación de búsqueda online golosa.
    Vamos "caminando" y descubriendo el grafo poquito a poquito.
    """
    actual = inicio
    camino_recorrido = [actual]

    # mapa_conocido guardará qué vecinos hemos descubierto de cada nodo
    mapa_conocido: Dict[str, List[str]] = {}
    visitados: Set[str] = set([actual])

    print("==============================================")
    print("TRACE Búsqueda Online (Greedy con memoria parcial)")
    print("==============================================\n")
    print(f"Inicio en {actual}, meta {meta}\n")

    for paso in range(MAX_PASOS):

        print(f"[Paso {paso}] Estoy en {actual}")
        print(f"  h({actual}) = {h(actual):.1f}")

        # ¿Llegamos a la meta?
        if actual == meta:
            print(f"  >> Llegué a la meta {meta}")
            break

        # "Sensar" vecinos reales desde el estado actual
        vecinos_reales = grafo_real[actual][:]  # copia
        mapa_conocido[actual] = vecinos_reales

        print(f"  Vecinos visibles ahora: {vecinos_reales}")
        for v in vecinos_reales:
            print(f"    h({v}) = {h(v):.1f}")

        # Elegir siguiente movimiento (greedy online)
        siguiente = elegir_siguiente(actual, vecinos_reales, visitados)
        print(f"  Decisión: ir a {siguiente} (h={h(siguiente):.1f})")

        # Actualizar estado
        actual = siguiente
        camino_recorrido.append(actual)
        visitados.add(actual)

        print("")

    print("==============================================")
    print("RESULTADO BÚSQUEDA ONLINE")
    print("==============================================")
    print("Camino recorrido:", " -> ".join(camino_recorrido))
    print("¿Llegamos a la meta?:", "Sí" if camino_recorrido[-1] == meta else "No")
    print("Nodos visitados    :", visitados)
    print("Mapa conocido (parcial):")
    for nodo, vecs in mapa_conocido.items():
        print(f"  {nodo} -> {vecs}")
    print("")

    return camino_recorrido


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    # Ejemplo: arrancamos en 'A' y queremos llegar a 'L'
    ruta = busqueda_online('A', 'L', MAX_PASOS=20)

    # Nota importante:
    # Esta búsqueda online es voraz y local.
    # No garantiza el camino más corto,
    # y puede quedarse atrapada rebotando localmente.
    #
    # Pero:
    # - Sí modela un agente que descubre el mundo paso a paso.
    # - Sí construye un pequeño "mapa_conocido".
    # - Sí deja registro del recorrido real que tomó.
