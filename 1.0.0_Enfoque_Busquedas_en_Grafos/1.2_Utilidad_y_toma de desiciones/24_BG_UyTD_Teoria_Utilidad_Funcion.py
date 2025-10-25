# =========================================================
# 24 - TEORÍA DE LA UTILIDAD: FUNCIÓN DE UTILIDAD
# ---------------------------------------------------------
# Descripción:
#   En este módulo modelamos y calculamos la utilidad
#   esperada (UE) de diferentes decisiones posibles.
#
#   Cada decisión lleva a resultados con probabilidades
#   y utilidades distintas. La meta es escoger la acción
#   que maximiza la utilidad esperada.
#
#   UE(a) = Σ [ P(resultado | acción) * U(resultado) ]
#
# Grafo de decisión:
#
#                 [ Decisión: Elegir Ruta ]
#                      /      |      \
#                   RutaA   RutaB   RutaC
#                    |        |       |
#             ┌──────┘   ┌───┘   ┌───┘
#             ▼          ▼       ▼
#       [ResultadoA] [ResultadoB] [ResultadoC]
#          p=1.0       p=0.8/0.2    p=0.6/0.4
#          U=50        U=80 / -100   U=60 / -40
#
# =========================================================

from typing import Dict, List

# ---------------------------------------------------------
# 1. Definimos las acciones y sus posibles resultados
# ---------------------------------------------------------
# Cada acción tiene una lista de tuplas (probabilidad, utilidad)
# ---------------------------------------------------------

acciones: Dict[str, List[tuple]] = {
    'RutaA': [(1.0, 50)],             # 100% éxito, utilidad = +50
    'RutaB': [(0.8, 80), (0.2, -100)],# 80% éxito +80, 20% fallo -100
    'RutaC': [(0.6, 60), (0.4, -40)]  # 60% éxito +60, 40% fallo -40
}

# ---------------------------------------------------------
# 2. Función para calcular la utilidad esperada
# ---------------------------------------------------------
def utilidad_esperada(resultados: List[tuple]) -> float:
    """
    Calcula la utilidad esperada de una lista de (prob, utilidad).
    """
    total = 0
    for p, u in resultados:
        total += p * u
    return total

# ---------------------------------------------------------
# 3. Evaluar cada acción
# ---------------------------------------------------------
def evaluar_acciones(acciones: Dict[str, List[tuple]]):
    print("==============================================")
    print("EVALUACIÓN DE UTILIDAD ESPERADA")
    print("==============================================\n")

    mejor_accion = None
    mejor_UE = float('-inf')

    for accion, outcomes in acciones.items():
        print(f"Acción: {accion}")
        for p, u in outcomes:
            print(f"   - P={p:.2f}  U={u}")
        UE = utilidad_esperada(outcomes)
        print(f"   -> Utilidad Esperada = {UE:.2f}\n")

        if UE > mejor_UE:
            mejor_UE = UE
            mejor_accion = accion

    print("==============================================")
    print(f"MEJOR DECISIÓN: {mejor_accion}")
    print(f"UTILIDAD ESPERADA MÁXIMA: {mejor_UE:.2f}")
    print("==============================================\n")

    return mejor_accion, mejor_UE

# ---------------------------------------------------------
# 4. MAIN: Ejemplo de ejecución
# ---------------------------------------------------------
if __name__ == "__main__":
    mejor, valor = evaluar_acciones(acciones)

    # Nota:
    # - La utilidad esperada combina beneficio y riesgo.
    # - Una acción con alta recompensa pero alta penalización
    #   puede tener menor utilidad esperada que una segura.
    #
    # - Aquí, la decisión racional es elegir la acción con
    #   mayor UE, no necesariamente la que da más recompensa
    #   individual.
