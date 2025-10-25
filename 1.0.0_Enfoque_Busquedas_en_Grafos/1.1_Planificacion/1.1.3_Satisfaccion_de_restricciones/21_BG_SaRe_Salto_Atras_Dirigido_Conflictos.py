# =========================================================
# 21 - SALTO ATRÁS DIRIGIDO POR CONFLICTOS
#     (Conflict-Directed Backjumping / Backjumping)
# ---------------------------------------------------------
# Descripción:
#   Este módulo implementa una versión educativa de
#   "Backjumping Dirigido por Conflictos".
#
#   La idea es mejorar el backtracking clásico (módulo 18):
#
#   En backtracking normal:
#       - Si una variable X no puede tomar ningún valor válido,
#         retrocedes solo a la variable anterior en el orden.
#
#   En backjumping:
#       - Cuando una variable X falla,
#         analizas CON QUIÉN tiene el conflicto directo.
#       - Y saltas ("jump") hacia la variable conflictiva más
#         relevante, en lugar de regresar solo un paso.
#
#   Esto ahorra exploración inútil cuando es obvio que
#   el problema viene de una decisión vieja, no de la más reciente.
#
# Modelo CSP:
#   - Variables:
#        ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
#   - Dominio:
#        { 'ROJO', 'VERDE', 'AZUL' }
#   - Restricciones binarias (vecinas ≠ mismo color)
#   - Problema: colorear regiones del mapa de Australia.
#
# Nota de implementación:
#   - Vamos a mantener un rastro de "conflict_set" por variable.
#   - conflict_set[X] = {variables anteriores que chocan con X}
#
#   Flujo básico:
#     1) Intento asignar variable[i].
#     2) Si no puedo asignar ningún valor consistente:
#           - construyo un conjunto de conflicto (quién la está bloqueando).
#           - hago 'jump' hacia la variable más "lejana" dentro de ese conjunto.
#
#   Esto es una simplificación didáctica del algoritmo formal,
#   pero mantiene la idea clave:
#   "retrocede hasta donde está el conflicto real".
#
# Trazas:
#   Imprimimos asignaciones, conflictos detectados,
#   y saltos de backjump.
#
# =========================================================

from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------
# 1. Definición del CSP (misma base que 17-20)
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
    """True si var_a=val_a y var_b=val_b NO violan la restricción."""
    if var_b not in adyacentes.get(var_a, []):
        return True
    return val_a != val_b


def es_consistente_con_parcial(asignacion: Dict[str, str],
                               var: str,
                               valor: str) -> Tuple[bool, Set[str]]:
    """
    Revisa si var=valor es consistente con la asignación parcial.
    Regresa:
      - booleano consistente / no consistente
      - conjunto de variables que causaron conflicto directo
    """
    conflictos = set()
    for otra_var, otro_val in asignacion.items():
        if not no_conflicto_binario(var, valor, otra_var, otro_val):
            conflictos.add(otra_var)
    return (len(conflictos) == 0, conflictos)


# ---------------------------------------------------------
# 2. Backjumping Dirigido por Conflictos
# ---------------------------------------------------------
def backjumping(
    orden_vars: List[str],
    dominios: Dict[str, List[str]]
) -> Optional[Dict[str, str]]:
    """
    Implementación educativa de Conflict-Directed Backjumping.

    Mantenemos:
      - asignacion_parcial: Dict[var] = valor
      - idx_actual: índice de la variable que estamos intentando asignar
      - conflict_set: Dict[var] = set(vars que le causaron conflicto)

    Proceso:
      - Intentamos asignar valores a la variable actual.
      - Si algún valor es consistente: avanzamos.
      - Si ningún valor sirve: hacemos "backjump":
           combinamos conflict_set[var_actual] con conflict_set de los
           que chocaron, y saltamos hacia la variable más lejana
           en ese conjunto de conflicto.
    """

    # Estado incremental de construcción
    asignacion_parcial: Dict[str, str] = {}

    # conflict_set[var] = {vars previas que impiden asignar var}
    conflict_set: Dict[str, Set[str]] = {var: set() for var in orden_vars}

    # Para cada variable, llevamos un índice de qué valor del dominio
    # estamos probando actualmente. Si agotamos todos, hay que backjump.
    intento_valor_idx: Dict[str, int] = {var: 0 for var in orden_vars}

    # Empezamos en la primera variable del orden
    idx_actual = 0

    while True:
        # Caso: si idx_actual es negativo, significa que saltamos
        # más allá del inicio -> no hay solución.
        if idx_actual < 0:
            print(">> No hay solución. Se retrocedió más allá del inicio.")
            return None

        # Caso: si idx_actual == len(orden_vars), ya asignamos todo.
        if idx_actual == len(orden_vars):
            print(">> ÉXITO: asignación completa encontrada")
            print("   ", asignacion_parcial, "\n")
            return asignacion_parcial

        var_actual = orden_vars[idx_actual]
        dominio_var = dominios[var_actual]

        print(f"[Nivel {idx_actual}] Variable actual: {var_actual}")
        print(f"  dominio[{var_actual}] = {dominio_var}")
        print(f"  conflict_set[{var_actual}] = {conflict_set[var_actual]}")
        print(f"  asignación parcial actual: {asignacion_parcial}")

        asignado_exitoso = False

        # Intentar valores desde intento_valor_idx[var_actual] en adelante
        while intento_valor_idx[var_actual] < len(dominio_var):
            valor = dominio_var[intento_valor_idx[var_actual]]
            print(f"    Probar {var_actual} = {valor} ...", end=" ")

            consistente, conflictos_directos = es_consistente_con_parcial(
                asignacion_parcial,
                var_actual,
                valor
            )

            if consistente:
                print("OK")

                # Asignamos este valor
                asignacion_parcial[var_actual] = valor

                # IMPORTANTE:
                # Resetear el índice de prueba de la siguiente variable
                if idx_actual + 1 < len(orden_vars):
                    intento_valor_idx[orden_vars[idx_actual + 1]] = 0

                asignado_exitoso = True
                break
            else:
                print(f"CONFLICTO con {conflictos_directos}")
                # Guardamos las variables que bloquearon este intento
                conflict_set[var_actual].update(conflictos_directos)

                # probamos el siguiente valor del dominio
                intento_valor_idx[var_actual] += 1

        if asignado_exitoso:
            # Avanzamos a la siguiente variable
            print(f"    -> {var_actual} queda asignada como {asignacion_parcial[var_actual]}\n")
            idx_actual += 1
            continue

        # Si llegamos aquí: no pudimos asignar NINGÚN valor a var_actual
        print(f"    !! Fallo total en {var_actual}, no hay valor posible.")

        # --- Backjumping ---
        # Si no pude asignar var_actual, necesito saltar hacia atrás,
        # pero NO necesariamente al idx_actual-1;
        # voy a saltar al índice más grande entre las variables
        # culpables en conflict_set[var_actual].

        if len(conflict_set[var_actual]) == 0:
            # No hay información útil de conflicto -> salto normal
            salto_idx = idx_actual - 1
            print(f"    No hay conjunto de conflicto. Salto normal a nivel {salto_idx}.\n")
        else:
            # Buscamos el índice más grande entre las variables en conflicto
            # (la más "profunda" en la asignación que está causando el problema)
            indices_conflictivos = [
                orden_vars.index(v) for v in conflict_set[var_actual]
                if v in orden_vars
            ]
            if len(indices_conflictivos) == 0:
                salto_idx = idx_actual - 1
            else:
                salto_idx = max(indices_conflictivos)

            print(f"    Conflicto causado por: {conflict_set[var_actual]}")
            print(f"    Saltaremos directamente a la variable en índice {salto_idx} ({orden_vars[salto_idx]})\n")

            # Antes de saltar, unimos la info de conflicto:
            # decimos que la variable a la que saltamos hereda los conflictos.
            # Esto es la parte "dirigida por conflicto":
            # si Z falló por culpa de {X,Y}, entonces X/Y deben saberlo.
            if salto_idx >= 0:
                conflict_set[orden_vars[salto_idx]].update(conflict_set[var_actual])

        # Limpiar la asignación de todas las variables desde idx_actual en adelante
        # porque vamos a reintentar desde el punto de salto.
        for i in range(idx_actual, len(orden_vars)):
            var_i = orden_vars[i]
            if var_i in asignacion_parcial:
                print(f"    (Desasignando {var_i} = {asignacion_parcial[var_i]})")
                del asignacion_parcial[var_i]
            # También reseteamos conflict_set para las que están después,
            # porque vamos a reconsiderarlas.
            if i != salto_idx:
                conflict_set[orden_vars[i]] = set()
            # Cuando volvamos a intentar esta variable más tarde,
            # queremos empezar probando el SIGUIENTE valor del dominio si aplica
            # excepto en el punto donde vamos a aterrizar ahora.
            if i != salto_idx:
                intento_valor_idx[orden_vars[i]] = 0

        # Ahora, importantísimo:
        # en la variable a la que vamos a saltar (salto_idx),
        # tenemos que avanzar su intento_valor_idx para probar
        # su "siguiente" valor distinto al que ya se intentó.
        if salto_idx >= 0:
            var_salto = orden_vars[salto_idx]
            intento_valor_idx[var_salto] += 1
            print(f"    Reintentaremos {var_salto} con otro valor (siguiente en su dominio).")
            print(f"    intento_valor_idx[{var_salto}] ahora = {intento_valor_idx[var_salto]}\n")

        # Actualizamos idx_actual al punto de salto
        idx_actual = salto_idx


# ---------------------------------------------------------
# 3. MAIN: Ejecutar Backjumping con trazas
# ---------------------------------------------------------
if __name__ == "__main__":
    print("==============================================")
    print("TRACE Salto Atrás Dirigido por Conflictos")
    print("==============================================\n")

    orden_variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']

    solucion = backjumping(
        orden_vars=orden_variables,
        dominios={var: dominio_base[var][:] for var in orden_variables}
    )

    print("==============================================")
    print("RESULTADO FINAL BACKJUMPING")
    print("==============================================")
    print("Asignación encontrada:", solucion)
    print("¿Solución completa?:", solucion is not None and len(solucion) == len(variables))
    print("")

    # Nota:
    # - Este algoritmo no explora ciegamente hacia atrás paso a paso.
    #   En lugar de eso, usa información de conflicto para "saltar"
    #   directamente a la decisión que realmente causó el problema.
    #
    # - Eso reduce trabajo comparado con backtracking puro,
    #   especialmente en CSP grandes y densos.
    #
    # - En el siguiente módulo (22),
    #   cambiamos completamente de paradigma:
    #   pasamos de búsqueda sistemática (árbol) a búsqueda local,
    #   usando el algoritmo de Mínimos-Conflictos.
