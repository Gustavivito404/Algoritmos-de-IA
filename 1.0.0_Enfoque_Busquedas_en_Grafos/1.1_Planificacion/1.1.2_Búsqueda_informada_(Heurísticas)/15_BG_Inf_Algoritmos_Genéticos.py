# =========================================================
# 15 - ALGORITMOS GENÉTICOS (Genetic Algorithms)
# ---------------------------------------------------------
# Descripción:
#   Los Algoritmos Genéticos (AG) son métodos de búsqueda
#   y optimización inspirados en la evolución biológica.
#
#   La idea es simular la selección natural:
#     - Cada posible solución es un "individuo".
#     - Cada individuo tiene un "genotipo" (cadena de genes).
#     - La calidad de una solución se mide con una función de
#       aptitud (fitness).
#     - La población evoluciona mediante:
#         * selección (los mejores sobreviven),
#         * cruce (recombinan sus genes),
#         * mutación (cambios aleatorios).
#
#   En este ejemplo, cada individuo representa un posible nodo
#   del grafo, y su aptitud depende de qué tan cerca está de la
#   meta 'L' según la heurística Manhattan.
#
# Heurística:
#   h(n) = |x_n - x_L| + |y_n - y_L|
#   fitness(n) = 1 / (1 + h(n))
#   (Mientras menor h, mayor fitness)
#
# Grafo usado (mismo que los anteriores):
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
# Coordenadas usadas:
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
# Flujo del algoritmo:
#
#   1. Crear una población inicial de N individuos (nodos aleatorios).
#   2. Evaluar su fitness.
#   3. Repetir durante MAX_GEN:
#        - Seleccionar los mejores individuos (por ruleta o ranking).
#        - Cruzar pares para generar nuevos hijos.
#        - Aplicar mutación aleatoria (cambiar a un vecino).
#        - Reemplazar población vieja con los nuevos individuos.
#        - Registrar el mejor individuo de cada generación.
#
#   4. Al final, devolvemos el mejor nodo (mayor fitness),
#      junto con la historia de la evolución.
#
# =========================================================

from typing import Dict, List, Tuple
from math import fabs
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
# Coordenadas (actualizadas)
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


# ---------------------------------------------------------
# Heurística y Fitness
# ---------------------------------------------------------
def heuristica_manhattan(nodo: str, meta: str = META) -> float:
    (x1, y1) = coords[nodo]
    (x2, y2) = coords[meta]
    return fabs(x1 - x2) + fabs(y1 - y2)


def fitness(nodo: str) -> float:
    """
    Fitness inversamente proporcional a h(n).
    Entre más cerca del objetivo, mayor fitness.
    """
    h_val = heuristica_manhattan(nodo)
    return 1 / (1 + h_val)


# ---------------------------------------------------------
# Selección por ruleta
# ---------------------------------------------------------
def seleccion_ruleta(poblacion: List[str], fitnesses: List[float]) -> str:
    total_fit = sum(fitnesses)
    pick = random.uniform(0, total_fit)
    acumulado = 0.0
    for i, f in enumerate(fitnesses):
        acumulado += f
        if acumulado >= pick:
            return poblacion[i]
    return poblacion[-1]


# ---------------------------------------------------------
# Cruce y mutación
# ---------------------------------------------------------
def cruzar(padre1: str, padre2: str) -> str:
    """
    Cruce simple: tomamos aleatoriamente un vecino de alguno de los padres.
    """
    if random.random() < 0.5:
        base = padre1
    else:
        base = padre2

    vecinos = graph[base]
    if vecinos:
        return random.choice(vecinos)
    else:
        return base


def mutar(nodo: str, prob_mutacion: float = 0.2) -> str:
    """
    Con cierta probabilidad, cambiamos a un vecino aleatorio.
    """
    if random.random() < prob_mutacion and graph[nodo]:
        return random.choice(graph[nodo])
    return nodo


# ---------------------------------------------------------
# Algoritmo Genético principal
# ---------------------------------------------------------
def algoritmo_genetico(
    TAM_POBLACION: int = 6,
    MAX_GEN: int = 10,
    PROB_MUTACION: float = 0.3
) -> Tuple[List[str], str, float]:
    """
    Algoritmo Genético básico adaptado al grafo:
    Cada nodo es un individuo, evaluado por su cercanía a L.
    """
    # Población inicial (nodos aleatorios del grafo)
    poblacion = random.sample(list(graph.keys()), TAM_POBLACION)
    mejor_global = poblacion[0]
    mejor_fit = fitness(mejor_global)

    print("==============================================")
    print("TRACE Algoritmo Genético")
    print("==============================================\n")
    print(f"Población inicial: {poblacion}\n")

    for gen in range(MAX_GEN):
        fits = [fitness(n) for n in poblacion]

        # Evaluar mejor individuo actual
        mejor_local = poblacion[fits.index(max(fits))]
        fit_local = max(fits)
        if fit_local > mejor_fit:
            mejor_fit = fit_local
            mejor_global = mejor_local

        print(f"[Gen {gen}]")
        for n, f in zip(poblacion, fits):
            print(f"  {n}: h={heuristica_manhattan(n):.1f}, fitness={f:.3f}")

        print(f"  Mejor local: {mejor_local} (fit={fit_local:.3f})")
        print(f"  Mejor global: {mejor_global} (fit={mejor_fit:.3f})\n")

        # Nueva generación
        nueva_poblacion = []
        for _ in range(TAM_POBLACION):
            padre1 = seleccion_ruleta(poblacion, fits)
            padre2 = seleccion_ruleta(poblacion, fits)
            hijo = cruzar(padre1, padre2)
            hijo = mutar(hijo, PROB_MUTACION)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion

    print("==============================================")
    print("RESULTADO FINAL - ALGORITMO GENÉTICO")
    print("==============================================")
    print("Población final:", poblacion)
    print("Mejor individuo global:", mejor_global)
    print("h(mejor):", heuristica_manhattan(mejor_global))
    print("Fitness(mejor):", mejor_fit)
    print("¿Es la meta 'L'?:", "Sí" if mejor_global == META else "No")
    print("")

    return poblacion, mejor_global, mejor_fit


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)

    poblacion_final, mejor, fit = algoritmo_genetico(
        TAM_POBLACION=6,
        MAX_GEN=10,
        PROB_MUTACION=0.3
    )

    print("Resumen final:")
    print("Población final:", poblacion_final)
    print("Mejor individuo:", mejor)
    print("Fitness:", fit)
