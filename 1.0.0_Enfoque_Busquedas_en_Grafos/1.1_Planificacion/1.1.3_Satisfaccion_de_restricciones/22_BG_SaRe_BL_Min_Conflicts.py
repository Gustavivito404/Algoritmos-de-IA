# =========================================================
# 22 - BÚSQUEDA LOCAL: MÍNIMOS-CONFLICTOS (Min-Conflicts)
# ---------------------------------------------------------
# Descripción:
#   Este módulo implementa el algoritmo de Mínimos-Conflictos
#   (Min-Conflicts), un enfoque de búsqueda LOCAL para CSP.
#
#   A diferencia de backtracking (18), forward checking (19),
#   propagación de restricciones (20) o backjumping (21),
#   aquí NO construimos la solución paso a paso.
#
#   En su lugar:
#     1. Arrancamos con una asignación COMPLETA inicial
#        (posiblemente con conflictos).
#     2. Mientras haya conflictos:
#          - Elegimos una variable que esté en conflicto.
#          - Le reasignamos el valor que genera MENOS conflictos
#            con sus vecinos.
#     3. Repetimos hasta que ya no haya conflictos o hasta
#        llegar a un número máximo de iteraciones.
#
#   Este método es MUY usado en problemas grandes de horarios,
#   asignación de recursos y planificación, porque en la práctica
#   suele llegar rápido a soluciones sin necesidad de retroceder.
#
# Modelo CSP usado (mismo que en 17-21):
#   • Variables:
#        ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
#   • Dominio:
#        { 'ROJO', 'VERDE', 'AZUL' }
#   • Restricciones binarias:
#        Si A y B son vecinas => color(A) != color(B)
#
#   Esto modela el coloreo del mapa de Australia: queremos
#   colorear cada región distinto a sus vecinas.
#
# Estrategia Min-Conflicts:
#   - Mantiene SIEMPRE una asignación completa (todas las regiones tienen color).
#   - No hace backtracking recursivo.
#   - Localmente repara las violaciones.
#
# Trazas:
#   - Imprimimos la asignación inicial aleatoria.
#   - En cada iteración mostramos:
#       * qué variable estaba causando conflicto,
#       * a qué valor la cambiamos,
#       * cuántos conflictos totales quedan.
#
# Nota:
#   Este algoritmo es aleatorio. Distintas corridas pueden
#   dar pasos diferentes, pero suelen converger.
#
# =========================================================

from typing import Dict, List, Tuple
import random

# ---------------------------------------------------------
# 1. Definición del CSP (misma base que 17-21)
# ---------------------------------------------------------

variables: List[str] = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']

dominio: Dict[str, List[str]] = {
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


def conflicto_binario(var_a: str, val_a: str,
                      var_b: str, val_b: str) -> bool:
    """
    True si var_a=val_a y var_b=val_b VIOLAN la restricción.
    (O sea: si son vecinas y tienen el MISMO color.)
    """
    if var_b not in adyacentes.get(var_a, []):
        return False
    return val_a == val_b


# ---------------------------------------------------------
# 2. Métricas de conflicto
# ---------------------------------------------------------
def contar_conflictos_variable(asignacion: Dict[str, str],
                               var: str) -> int:
    """
    Cuenta cuántos conflictos causa 'var' con sus vecinos.
    """
    color_var = asignacion[var]
    conflictos = 0
    for vecino in adyacentes[var]:
        if vecino in asignacion:
            if conflicto_binario(var, color_var, vecino, asignacion[vecino]):
                conflictos += 1
    return conflictos


def contar_conflictos_totales(asignacion: Dict[str, str]) -> int:
    """
    Cuenta el número total de conflictos en toda la asignación.
    NOTA: Para no duplicar, solo contamos (A,B) si A < B en orden.
    """
    total = 0
    for i in range(len(variables)):
        for j in range(i + 1, len(variables)):
            vi = variables[i]
            vj = variables[j]
            if conflicto_binario(vi, asignacion[vi], vj, asignacion[vj]):
                total += 1
    return total


# ---------------------------------------------------------
# 3. Construcción de una asignación inicial completa
# ---------------------------------------------------------
def asignacion_inicial_aleatoria() -> Dict[str, str]:
    """
    Asigna a cada variable un color aleatorio de su dominio.
    Esta asignación puede tener conflictos al inicio.
    """
    return {
        var: random.choice(dominio[var])
        for var in variables
    }


# ---------------------------------------------------------
# 4. Elegir variable en conflicto y repararla
# ---------------------------------------------------------
def elegir_variable_en_conflicto(asignacion: Dict[str, str]) -> str:
    """
    Elige una variable que esté en conflicto con al menos un vecino.
    Si hay varias, elige una al azar entre las conflictivas.
    """
    conflictivas = [
        var for var in variables
        if contar_conflictos_variable(asignacion, var) > 0
    ]
    return random.choice(conflictivas) if conflictivas else None


def mejor_color_para_variable(asignacion: Dict[str, str],
                              var: str) -> Tuple[str, int]:
    """
    Para una variable dada, probamos todos los colores posibles
    en su dominio y medimos cuántos conflictos produciría cada uno.
    Elegimos el color que minimiza los conflictos.
    Regresa (color_elegido, conflictos_resultantes).
    """
    mejor_color = None
    mejor_conflictos = None

    for color in dominio[var]:
        # asignación hipotética temporal
        color_original = asignacion[var]
        asignacion[var] = color

        conflictos_actuales = contar_conflictos_variable(asignacion, var)

        # restaurar
        asignacion[var] = color_original

        if (mejor_conflictos is None) or (conflictos_actuales < mejor_conflictos):
            mejor_conflictos = conflictos_actuales
            mejor_color = color

    return mejor_color, mejor_conflictos


# ---------------------------------------------------------
# 5. Algoritmo Min-Conflicts
# ---------------------------------------------------------
def min_conflicts(
    MAX_ITER: int = 50
) -> Dict[str, str]:
    """
    Algoritmo de búsqueda local Min-Conflicts.
    - Empieza con una asignación completa aleatoria.
    - Repite hasta MAX_ITER:
        * Si no hay conflictos -> devolvemos solución.
        * Escogemos una variable conflictiva.
        * Le ponemos el color que minimiza conflictos.
    - Si no logramos una solución sin conflictos en MAX_ITER,
      devolvemos la mejor asignación observada (la menos conflictiva
      que vimos durante la búsqueda).
    """

    asignacion = asignacion_inicial_aleatoria()
    mejor_global = dict(asignacion)
    mejor_global_conf = contar_conflictos_totales(asignacion)

    print("==============================================")
    print("TRACE Min-Conflicts (Búsqueda Local)")
    print("==============================================\n")

    print("Asignación inicial aleatoria:")
    print(asignacion)
    print("Conflictos iniciales:", mejor_global_conf)
    print("")

    for iter_idx in range(MAX_ITER):

        conflictos_totales = contar_conflictos_totales(asignacion)

        print(f"[Iter {iter_idx}] Conflictos totales actuales: {conflictos_totales}")
        print(f"  Asignación actual: {asignacion}")

        # ¿Ya no hay conflictos? -> solución válida
        if conflictos_totales == 0:
            print("  >> Sin conflictos. ¡Solución encontrada!\n")
            return asignacion

        # Mantener mejor solución vista hasta ahora
        if conflictos_totales < mejor_global_conf:
            mejor_global = dict(asignacion)
            mejor_global_conf = conflictos_totales
            print(f"  * Nuevo mejor global parcial ({mejor_global_conf} conflictos)")

        # 1) Elegimos una variable conflictiva
        var_prob = elegir_variable_en_conflicto(asignacion)
        print(f"  Variable conflictiva elegida: {var_prob}")

        # 2) Buscamos el mejor color para esa variable
        color_elegido, conflictos_resultantes = mejor_color_para_variable(asignacion, var_prob)
        print(f"  Mejor color para {var_prob}: {color_elegido} (conflictos locales -> {conflictos_resultantes})")

        # 3) Reasignamos ese color
        asignacion[var_prob] = color_elegido
        print(f"  Reasignado {var_prob} = {color_elegido}\n")

    print("==============================================")
    print("AVISO: No se alcanzó estado sin conflictos dentro de MAX_ITER")
    print("Devolviendo la mejor asignación parcial encontrada.")
    print("Mejor asignación parcial:", mejor_global)
    print("Conflictos en la mejor asignación parcial:", mejor_global_conf)
    print("")
    return mejor_global


# ---------------------------------------------------------
# 6. MAIN: Ejecutar Min-Conflicts con trazas
# ---------------------------------------------------------
if __name__ == "__main__":
    # Para tener reproducibilidad al estudiar la traza.
    random.seed(42)

    solucion = min_conflicts(MAX_ITER=50)

    print("==============================================")
    print("RESULTADO FINAL MIN-CONFLICTS")
    print("==============================================")
    print("Asignación final:", solucion)
    print("Conflictos finales:", contar_conflictos_totales(solucion))
    print("¿Es solución válida (0 conflictos)?",
          contar_conflictos_totales(solucion) == 0)
    print("")

    # Nota:
    # - Min-Conflicts trabaja con asignaciones completas
    #   y hace "reparaciones locales".
    #
    # - Es muy eficiente en problemas grandes porque,
    #   en vez de explorar rutas en árbol como backtracking,
    #   camina directamente por el espacio de soluciones completas.
    #
    # - En el módulo 23 vamos a ver "Acondicionamiento del Corte",
    #   que es otra idea potente pero más estructural:
    #   romper el problema en partes más fáciles fijando
    #   ciertas variables clave.
