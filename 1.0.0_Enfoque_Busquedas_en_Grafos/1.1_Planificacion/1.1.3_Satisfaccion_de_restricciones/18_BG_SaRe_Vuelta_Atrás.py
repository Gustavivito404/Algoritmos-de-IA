# =========================================================
# 18 - BÚSQUEDA DE VUELTA ATRÁS (Backtracking Search)
# ---------------------------------------------------------
# Descripción:
#   En este módulo implementamos la búsqueda de vuelta atrás
#   (backtracking) clásica para resolver un CSP.
#
#   Objetivo:
#     Asignar un color a cada región del mapa (WA, NT, SA, ...)
#     tal que regiones vecinas no tengan el mismo color.
#
#   En backtracking puro:
#     - Asignamos variables una por una.
#     - Probamos valores del dominio en orden.
#     - Si una asignación parcial viola una restricción,
#       retrocedemos ("backtrack") y probamos otro valor.
#
#   Esto es básicamente una búsqueda en profundidad con poda
#   temprana por restricciones.
#
# Características:
#   • No hay heurísticas de orden inteligente de variables.
#   • No hay poda extra de dominios (eso vendrá en 19 y 20).
#   • Solo revisa consistencia local en cada paso.
#
#   IMPORTANTE:
#   - Este algoritmo garantiza encontrar una solución válida
#     si existe, porque explora sistemáticamente.
#   - Pero puede ser lento en problemas grandes.
#
# Modelo CSP usado:
#   - Variables:
#        ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
#   - Dominio:
#        { 'ROJO', 'VERDE', 'AZUL' }
#   - Restricciones binarias:
#        Si A y B son vecinas => color(A) != color(B)
#
#   Esto modela el coloreo de regiones del mapa de Australia.
#
# Trazas:
#   Vamos a imprimir paso a paso:
#      - qué variable estoy tratando de asignar,
#      - qué valor intento,
#      - si hay conflicto o no,
#      - cuándo hago backtrack.
#
# =========================================================

from typing import Dict, List

# ---------------------------------------------------------
# 1. Definición del CSP (idéntica al módulo 17)
# ---------------------------------------------------------

variables: List[str] = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']

dominio_base: Dict[str, List[str]] = {
    var: ['ROJO', 'VERDE', 'AZUL'] for var in variables
}

adyacentes: Dict[str, List[str]] = {
    'WA':  ['NT', 'SA'],
    'NT':  ['WA', 'SA', 'Q'],
    'SA':  ['WA', 'NT', 'Q', 'NSW', 'V'],
    'Q':   ['NT', 'SA', 'NSW'],
    'NSW': ['SA', 'Q', 'V'],
    'V':   ['SA', 'NSW', 'T'],
    'T':   ['V']
}

def no_conflicto_binario(var_a: str, val_a: str,
                         var_b: str, val_b: str) -> bool:
    if var_b not in adyacentes.get(var_a, []):
        return True
    return val_a != val_b

def es_consistente_con_asignacion(actual_asignacion: Dict[str, str],
                                  var: str,
                                  valor: str) -> bool:
    """
    Revisa si puedo asignar var=valor sin violar restricciones
    con las variables YA asignadas.
    """
    for otra_var, otro_valor in actual_asignacion.items():
        if not no_conflicto_binario(var, valor, otra_var, otro_valor):
            return False
    return True


# ---------------------------------------------------------
# 2. Backtracking Search
# ---------------------------------------------------------
def backtracking(asignacion_parcial: Dict[str, str],
                 variables_orden: List[str],
                 dominios: Dict[str, List[str]],
                 nivel: int = 0) -> Dict[str, str]:
    """
    Intentamos construir una asignación completa válida.
    - asignacion_parcial: dict con las variables ya asignadas
    - variables_orden: orden fijo en el que asignaremos
    - dominios: valores posibles por variable
    - nivel: para identar la traza visual

    Retorna:
      - Una asignación completa válida si existe.
      - None (implícito) si falla.
    """

    # Caso base: ¿ya asignamos todas las variables?
    if len(asignacion_parcial) == len(variables_orden):
        print(" " * (2 * nivel) + f">> ÉXITO: asignación completa encontrada")
        print(" " * (2 * nivel) + f"   {asignacion_parcial}\n")
        return asignacion_parcial

    # Elegimos la siguiente variable sin asignar
    var_actual = variables_orden[len(asignacion_parcial)]

    print(" " * (2 * nivel) + f"- Intentando asignar variable: {var_actual}")

    # Probamos los valores del dominio en orden
    for valor in dominios[var_actual]:
        print(" " * (2 * nivel) + f"  Probar {var_actual} = {valor} ...", end=" ")

        # Checamos si es consistente con lo ya asignado
        if es_consistente_con_asignacion(asignacion_parcial, var_actual, valor):
            print("OK")
            # Hacemos la asignación tentativa
            asignacion_parcial[var_actual] = valor

            # Llamada recursiva
            resultado = backtracking(asignacion_parcial,
                                     variables_orden,
                                     dominios,
                                     nivel + 1)
            if resultado is not None:
                return resultado  # propagamos el éxito hacia arriba

            # Si no funcionó, quitamos y probamos siguiente valor
            print(" " * (2 * nivel) + f"  (Backtrack) Retiro {var_actual} = {valor}")
            del asignacion_parcial[var_actual]
        else:
            print("CONFLICTO")

    # Si ningún valor funcionó para esta variable, fallo en este punto
    print(" " * (2 * nivel) + f"!! Fallo en {var_actual}, regreso nivel arriba\n")
    return None


# ---------------------------------------------------------
# 3. MAIN: Ejecutar el backtracking con trazas
# ---------------------------------------------------------
if __name__ == "__main__":
    print("==============================================")
    print("TRACE Backtracking Search (Búsqueda de Vuelta Atrás)")
    print("==============================================\n")

    # Orden fijo de las variables (versión simple)
    orden_variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']

    solucion = backtracking(
        asignacion_parcial={},
        variables_orden=orden_variables,
        dominios=dominio_base
    )

    print("==============================================")
    print("RESULTADO FINAL BACKTRACKING")
    print("==============================================")
    print("Asignación encontrada:", solucion)
    print("¿Es solución completa?:", solucion is not None and len(solucion) == len(variables))
    print("")

    # Nota:
    # - Esta versión es el backtracking "puro":
    #     • No hace recorte de dominios futuro.
    #     • No elige la siguiente variable inteligentemente.
    #   Por eso puede explorar mucho.
    #
    # - En el módulo 19 (Comprobación Hacia Delante)
    #   vamos a mejorar esto reduciendo dominios de las
    #   variables NO asignadas inmediatamente después
    #   de cada decisión para detectar callejones pronto.