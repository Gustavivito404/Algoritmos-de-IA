# =========================================================
# 25 - REDES DE DECISIÓN (Decision Networks)
# ---------------------------------------------------------
# Descripción:
#   Extendemos el modelo de utilidad esperada con una variable
#   incierta (Clima) que afecta la probabilidad de éxito de cada
#   ruta. Esto nos permite representar dependencias condicionales
#   entre decisiones, estados del mundo y utilidades.
#
#   Calcularemos la UTILIDAD ESPERADA de cada decisión
#   considerando P(Clima), P(Éxito | Ruta, Clima) y U(Éxito).
#
# Grafo de decisión:
#
#          [ Clima ]
#             │
#             ▼
#         [ Decisión ]
#        /     |      \
#     RutaA  RutaB   RutaC
#       │       │       │
#       ▼       ▼       ▼
#    [ Éxito ] [ Éxito ] [ Éxito ]
#       │        │        │
#       └────────┴────────┘
#                │
#                ▼
#            [ Utilidad ]
#
# =========================================================

from typing import Dict, List

# ---------------------------------------------------------
# 1. Definición de las distribuciones de probabilidad
# ---------------------------------------------------------

# Probabilidades del clima
P_clima = {
    'Soleado': 0.7,
    'Lluvioso': 0.3
}

# Probabilidad de éxito según la ruta y el clima
# P(Éxito=True | Ruta, Clima)
P_exito = {
    'RutaA': {'Soleado': 1.0,  'Lluvioso': 0.8},
    'RutaB': {'Soleado': 0.9,  'Lluvioso': 0.5},
    'RutaC': {'Soleado': 0.7,  'Lluvioso': 0.4}
}

# ---------------------------------------------------------
# 2. Función de utilidad (U)
# ---------------------------------------------------------
# Si el viaje tiene éxito -> +80 puntos de utilidad
# Si falla -> -100 puntos
U = {
    'Exito': 80,
    'Fallo': -100
}

# ---------------------------------------------------------
# 3. Cálculo de utilidad esperada por ruta
# ---------------------------------------------------------
def utilidad_esperada_ruta(ruta: str) -> float:
    """
    Calcula la utilidad esperada considerando:
      - Probabilidad del clima
      - Probabilidad de éxito condicional al clima y ruta
      - Utilidad de éxito/fallo
    """
    UE_total = 0.0
    for clima, p_clima in P_clima.items():
        p_exito = P_exito[ruta][clima]
        p_fallo = 1 - p_exito

        # utilidad esperada condicionada al clima
        UE_clima = (p_exito * U['Exito']) + (p_fallo * U['Fallo'])

        print(f"  [{ruta}] Clima={clima:<8} | P(clima)={p_clima:.2f} | P(éxito)={p_exito:.2f} | UE_clima={UE_clima:.2f}")

        # utilidad esperada marginalizada sobre el clima
        UE_total += p_clima * UE_clima

    return UE_total

# ---------------------------------------------------------
# 4. Evaluar todas las rutas
# ---------------------------------------------------------
def evaluar_rutas():
    print("==============================================")
    print("EVALUACIÓN DE RED DE DECISIÓN")
    print("==============================================\n")

    mejor_ruta = None
    mejor_UE = float('-inf')

    for ruta in ['RutaA', 'RutaB', 'RutaC']:
        print(f"Ruta evaluada: {ruta}")
        UE = utilidad_esperada_ruta(ruta)
        print(f"  → Utilidad Esperada Global({ruta}) = {UE:.2f}\n")

        if UE > mejor_UE:
            mejor_UE = UE
            mejor_ruta = ruta

    print("==============================================")
    print(f"MEJOR DECISIÓN: {mejor_ruta}")
    print(f"UTILIDAD ESPERADA MÁXIMA: {mejor_UE:.2f}")
    print("==============================================\n")

# ---------------------------------------------------------
# 5. MAIN: Ejemplo de ejecución
# ---------------------------------------------------------
if __name__ == "__main__":
    evaluar_rutas()

    # Nota:
    # - Esta red introduce incertidumbre explícita (Clima)
    #   y permite calcular decisiones racionales bajo riesgo.
    #
    # - Es una versión simple de una "Red de Influencia",
    #   que une Probabilidad + Decisión + Utilidad.
    #
    # - En el siguiente módulo (26),
    #   estudiaremos el "Valor de la Información":
    #   cuánto vale observar el clima antes de decidir la ruta.
