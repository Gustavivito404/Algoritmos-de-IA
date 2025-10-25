# =========================================================
# 13 - TEMPLE SIMULADO (Simulated Annealing)
# ---------------------------------------------------------
# Descripción:
#   Temple Simulado es una búsqueda local probabilística.
#   Sirve para escapar de óptimos locales aceptando, a veces,
#   movimientos "peores" a propósito. Al inicio es flexible,
#   luego se vuelve más estricto.
#
#   Se inspira en el proceso físico de enfriar metal fundido:
#   alta temperatura -> los átomos se mueven libremente,
#   baja temperatura -> el material se solidifica en una forma estable.
#
# Idea básica:
#   - Tenemos un estado actual (un nodo del grafo).
#   - Tenemos una función de calidad (queremos minimizar h(n)).
#   - En cada iteración movemos a un vecino:
#       * Si es MEJOR (h baja), lo aceptamos siempre.
#       * Si es PEOR (h sube), tal vez lo aceptamos con una
#         cierta probabilidad que depende de la temperatura T.
#
#   prob_aceptar_empeoramiento = exp( -Δh / T )
#
#   Donde:
#       Δh = h_nuevo - h_actual
#       T = temperatura actual (empieza alta, va bajando)
#
#   Cuando T es alta:
#       Aceptamos saltos malos más fácil -> exploración global.
#
#   Cuando T es baja:
#       Casi no aceptamos movimientos malos -> refinamiento local.
#
# Características:
#   - No necesita memoria tabú.
#   - A diferencia de Hill Climbing, puede "bajarse de la colina"
#     temporalmente y luego subir una mejor.
#
# Heurística usada:
#   h(n) = distancia Manhattan hasta la meta 'L'
#
# Grafo usado:
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
# Conexiones reales:
#   A: B,C
#   B: A,D,E
#   C: A,F,M
#   D: B,H,I
#   E: B,F,G
#   F: C,E,K,M
#   G: E,J
#   H: D
#   I: D,J
#   J: G,I,L
#   K: F,L
#   L: K,J
#   M: C,F,N
#   N: M
#
# Coordenadas (para Manhattan):
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
# Flujo del algoritmo (esta versión educativa):
#
#   1. Elegimos un nodo inicial.
#   2. Para iter in [0..MAX_ITER]:
#        - Calculamos h(actual).
#        - Elegimos un vecino aleatorio.
#        - Calculamos h(vecino).
#        - Si h(vecino) < h(actual): lo tomamos.
#          else:
#             lo tomamos con prob = exp(-(h_vec - h_act)/T)
#        - Bajamos la temperatura T = T * enfriamiento
#
#   Guardamos:
#       • el mejor nodo global encontrado (menor h)
#       • el historial de saltos y decisiones
#
# Nota:
#   Este script usa randomness. Cada corrida puede dar un
#   camino diferente. Es normal.
#
# =========================================================

from typing import Dict, List, Tuple
from math import fabs, exp
import random

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
# Coordenadas ACTUALIZADAS (tu versión con separación)
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
    Mientras más chico h(n), "más cerca" está ese nodo de la meta.
    """
    (x1, y1) = coords[nodo]
    (x2, y2) = coords[meta]
    return fabs(x1 - x2) + fabs(y1 - y2)


def simulated_annealing(
    inicio: str,
    meta: str,
    T_inicial: float = 5.0,
    enfriamiento: float = 0.9,
    MAX_ITER: int = 20
) -> Tuple[List[str], str, float]:
    """
    Temple Simulado (versión simple).
    - inicio: nodo inicial
    - meta: nodo objetivo (solo para evaluar h respecto a meta)
    - T_inicial: temperatura inicial
    - enfriamiento: factor multiplicativo (<1) para bajar T
    - MAX_ITER: cuántas iteraciones vamos a intentar

    Devuelve:
      - historial: nodos visitados en orden
      - mejor_global: el nodo más prometedor encontrado (menor h)
      - h_mejor_global: valor h de ese mejor nodo
    """

    actual = inicio
    h_actual = heuristica_manhattan(actual, meta)

    mejor_global = actual
    h_mejor_global = h_actual

    historial = [actual]

    T = T_inicial  # temperatura actual

    print("==============================================")
    print("TRACE Simulated Annealing (Temple Simulado)")
    print("==============================================\n")
    print(f"Inicio en {actual} con h={h_actual:.1f}, T inicial={T:.2f}")
    print(f"Meta deseada: {meta}\n")

    for paso in range(MAX_ITER):
        if actual == meta:
            print(f">> META alcanzada exactamente ({meta}) en iteración {paso}")
            break

        # Elegimos un vecino aleatorio del estado actual
        vecinos = graph[actual]
        vecino_candidato = random.choice(vecinos)

        h_vecino = heuristica_manhattan(vecino_candidato, meta)
        delta_h = h_vecino - h_actual  # cambio en la "calidad"

        print(f"[Iter {paso}] Nodo actual: {actual}")
        print(f"  h(actual) = {h_actual:.1f}")
        print(f"  T = {T:.4f}")
        print(f"  Vecino candidato: {vecino_candidato}  h={h_vecino:.1f}  Δh={delta_h:.1f}")

        # Regla de aceptación:
        # - Si mejora (delta_h < 0) => acepto directo
        # - Si empeora (delta_h >= 0) => acepto con probabilidad exp(-Δh / T)
        if delta_h < 0:
            aceptar = True
            razon = "Mejora directa (h baja)"
        else:
            # evitar división por cero si T está demasiado baja
            if T > 1e-8:
                prob = exp(-delta_h / T)
            else:
                prob = 0.0
            rand_val = random.random()
            aceptar = rand_val < prob
            razon = f"Empeora, prob={prob:.4f}, rand={rand_val:.4f}"

        print(f"  ¿Aceptar movimiento?: {aceptar} ({razon})")

        # Si aceptamos, nos movemos
        if aceptar:
            actual = vecino_candidato
            h_actual = h_vecino
            historial.append(actual)

            # ¿Es el mejor global que hemos visto?
            if h_actual < h_mejor_global:
                h_mejor_global = h_actual
                mejor_global = actual
                print(f"  * Nuevo mejor global: {mejor_global} con h={h_mejor_global:.1f}")

        # Enfriamos la temperatura
        T = T * enfriamiento

        print("")

    print("==============================================")
    print("RESULTADO TEMPLE SIMULADO")
    print("==============================================")
    print("Historial recorrido :", " -> ".join(historial))
    print("Mejor nodo global   :", mejor_global)
    print("h(mejor nodo global):", h_mejor_global)
    print("¿Llegamos a la meta exactamente?:", "Sí" if historial[-1] == meta else "No")
    print("")

    return historial, mejor_global, h_mejor_global


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    # Semilla fija para que la traza sea repetible al estudiar.
    # Si quieres comportamiento distinto cada corrida, comenta esta línea.
    random.seed(42)

    trayectoria, mejor_nodo, mejor_h = simulated_annealing(
        inicio='A',
        meta='L',
        T_inicial=5.0,      # temperatura inicial alta = más aleatorio
        enfriamiento=0.9,   # qué tan rápido enfría
        MAX_ITER=50         # pasos máximos
    )

    print("Resumen final:")
    print("Trayectoria:", " -> ".join(trayectoria))
    print("Mejor nodo global:", mejor_nodo)
    print("Heurística en mejor nodo:", mejor_h)
    print("Nota: Temple Simulado puede aceptar pasos 'malos' al inicio")
    print("      para escapar de óptimos locales, y luego se vuelve")
    print("      más estricto a medida que baja la temperatura.\n")
