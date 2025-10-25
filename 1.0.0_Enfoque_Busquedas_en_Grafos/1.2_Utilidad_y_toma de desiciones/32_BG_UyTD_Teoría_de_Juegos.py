# =========================================================
# 32 - TEORÍA DE JUEGOS: EQUILIBRIOS Y MECANISMOS
# ---------------------------------------------------------
# Descripción:
#   Simulación simple del Dilema del Prisionero.
#   Calcula el equilibrio de Nash para dos jugadores racionales.
#
# =========================================================

import numpy as np
import itertools

# ---------------------------------------------------------
# 1. Definimos jugadores y estrategias
# ---------------------------------------------------------
jugadores = ["A", "B"]
estrategias = ["Cooperar", "Traicionar"]

# Matriz de pagos:
# Cada celda es (pago_A, pago_B)
# orden: [A_Cooperar, A_Traicionar] × [B_Cooperar, B_Traicionar]

pagos = {
    ("Cooperar", "Cooperar"): (3, 3),
    ("Cooperar", "Traicionar"): (0, 5),
    ("Traicionar", "Cooperar"): (5, 0),
    ("Traicionar", "Traicionar"): (1, 1)
}

# ---------------------------------------------------------
# 2. Función para verificar si un perfil es Equilibrio de Nash
# ---------------------------------------------------------
def es_equilibrio_nash(estrategia_A, estrategia_B):
    # Pagos actuales
    pago_A, pago_B = pagos[(estrategia_A, estrategia_B)]

    # Si A cambiara de estrategia, ¿le iría mejor?
    for alt_A in estrategias:
        if alt_A != estrategia_A:
            pago_alt, _ = pagos[(alt_A, estrategia_B)]
            if pago_alt > pago_A:
                return False  # A tiene incentivo a desviarse

    # Si B cambiara de estrategia, ¿le iría mejor?
    for alt_B in estrategias:
        if alt_B != estrategia_B:
            _, pago_alt = pagos[(estrategia_A, alt_B)]
            if pago_alt > pago_B:
                return False  # B tiene incentivo a desviarse

    return True

# ---------------------------------------------------------
# 3. Búsqueda de Equilibrios de Nash
# ---------------------------------------------------------
equilibrios = []
for (a, b) in itertools.product(estrategias, repeat=2):
    if es_equilibrio_nash(a, b):
        equilibrios.append((a, b, pagos[(a, b)]))

# ---------------------------------------------------------
# 4. Resultados
# ---------------------------------------------------------
print("==============================================")
print("TRACE TEORÍA DE JUEGOS: DILEMA DEL PRISIONERO")
print("==============================================\n")

print("Estrategias posibles:")
for (a, b), (pa, pb) in pagos.items():
    print(f"A={a:10s}, B={b:10s}  ->  Pago(A)={pa}, Pago(B)={pb}")
print("")

if equilibrios:
    print("Equilibrios de Nash encontrados:")
    for e in equilibrios:
        print(f"  Estrategia: A={e[0]}, B={e[1]}  ->  Pagos={e[2]}")
else:
    print("No se encontraron equilibrios puros.")

# Nota:
# - En este juego el equilibrio de Nash es (Traicionar, Traicionar)
# - Aunque no es el óptimo colectivo (3,3), es estable:
#   nadie mejora desviándose por sí solo.
#
# - Este principio se usa en economía, IA multiagente y
#   diseño de mecanismos (auction theory, social choice, etc.).
