# =========================================================
# 26 - VALOR DE LA INFORMACIÓN
# ---------------------------------------------------------
# Descripción:
#   Calculamos el Valor Esperado de la Información Perfecta (VEIP)
#   para una variable incierta (Clima) que afecta la elección de ruta.
#
#   Comparamos dos escenarios:
#     1. Sin información del clima (decisión directa).
#     2. Con información perfecta del clima (sabemos si está
#        soleado o lluvioso antes de decidir).
#
#   VEIP = EU_con_info - EU_sin_info
#
# =========================================================

from typing import Dict

# ---------------------------------------------------------
# 1. Distribuciones base (del módulo anterior)
# ---------------------------------------------------------
P_clima = {
    'Soleado': 0.7,
    'Lluvioso': 0.3
}

P_exito = {
    'RutaA': {'Soleado': 1.0,  'Lluvioso': 0.8},
    'RutaB': {'Soleado': 0.9,  'Lluvioso': 0.5},
    'RutaC': {'Soleado': 0.7,  'Lluvioso': 0.4}
}

U = {'Exito': 80, 'Fallo': -100}


# ---------------------------------------------------------
# 2. Función de utilidad esperada condicional al clima
# ---------------------------------------------------------
def utilidad_esperada_ruta_dado_clima(ruta: str, clima: str) -> float:
    p_exito = P_exito[ruta][clima]
    p_fallo = 1 - p_exito
    UE = (p_exito * U['Exito']) + (p_fallo * U['Fallo'])
    return UE


# ---------------------------------------------------------
# 3. Escenario 1: SIN información (decisión directa)
# ---------------------------------------------------------
def utilidad_esperada_sin_info() -> float:
    """
    Calcula la utilidad esperada óptima cuando NO conocemos el clima.
    (Elegimos la ruta que tiene mayor utilidad esperada marginal.)
    """
    mejores = {}
    for ruta in ['RutaA', 'RutaB', 'RutaC']:
        UE_total = 0
        for clima, p_c in P_clima.items():
            UE_total += p_c * utilidad_esperada_ruta_dado_clima(ruta, clima)
        mejores[ruta] = UE_total

    ruta_opt = max(mejores, key=mejores.get)
    print(f"Mejor ruta sin información: {ruta_opt}, UE = {mejores[ruta_opt]:.2f}")
    return mejores[ruta_opt]


# ---------------------------------------------------------
# 4. Escenario 2: CON información perfecta del clima
# ---------------------------------------------------------
def utilidad_esperada_con_info() -> float:
    """
    Calcula la utilidad esperada si pudiéramos observar el clima
    antes de decidir la ruta.
    """
    EU_con_info = 0.0
    for clima, p_c in P_clima.items():
        # Para este clima, elegimos la mejor ruta
        mejores = {}
        for ruta in ['RutaA', 'RutaB', 'RutaC']:
            mejores[ruta] = utilidad_esperada_ruta_dado_clima(ruta, clima)
        mejor_ruta = max(mejores, key=mejores.get)
        mejor_UE = mejores[mejor_ruta]
        print(f"  Si el clima es {clima:<8}, mejor ruta = {mejor_ruta}, UE = {mejor_UE:.2f}")
        EU_con_info += p_c * mejor_UE
    return EU_con_info


# ---------------------------------------------------------
# 5. MAIN: Cálculo del VEIP
# ---------------------------------------------------------
if __name__ == "__main__":
    print("==============================================")
    print("VALOR DE LA INFORMACIÓN (VEIP)")
    print("==============================================\n")

    UE_sin = utilidad_esperada_sin_info()
    print("")
    UE_con = utilidad_esperada_con_info()
    print("")

    VEIP = UE_con - UE_sin

    print("==============================================")
    print(f"UE sin información : {UE_sin:.2f}")
    print(f"UE con información  : {UE_con:.2f}")
    print(f"Valor de la Información Perfecta (VEIP): {VEIP:.2f}")
    print("==============================================\n")

    # Nota:
    # - VEIP > 0 indica que la información tiene valor.
    # - Si VEIP = 0, la información no cambia la decisión óptima.
    # - En la práctica, el agente solo debería pagar por información
    #   si su costo es menor que el VEIP calculado.