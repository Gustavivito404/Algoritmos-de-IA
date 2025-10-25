# =========================================================
# 19 - COMPROBACIÓN HACIA DELANTE (Forward Checking)
# ---------------------------------------------------------
# Descripción:
#   Este módulo implementa "Comprobación Hacia Delante"
#   (Forward Checking) para CSP.
#
#   Es una mejora directa sobre la Búsqueda de Vuelta Atrás
#   (Backtracking) del módulo 18.
#
#   Idea:
#     Cuando asigno una variable X = valor,
#     miro a todas las variables vecinas NO asignadas
#     y elimino de sus dominios cualquier valor que
#     cause conflicto con X.
#
#     Si alguna variable vecina se queda SIN valores posibles,
#     sé inmediatamente que esta elección va a fallar,
#     y hago backtrack sin seguir expandiendo.
#
#   Esto reduce muchísimo el árbol de búsqueda porque
#   detecta callejones sin salida temprano.
#
# Modelo CSP:
#   - Variables:
#        ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
#   - Dominio:
#        { 'ROJO', 'VERDE', 'AZUL' }
#   - Restricciones binarias:
#        Si A y B son vecinas => color(A) != color(B)
#
#   Estamos modelando el coloreo del mapa de Australia.
#
# Diferencia con módulo 18 (Backtracking):
#   - Backtracking solo revisa consistencia con lo que YA está asignado.
#   - Forward Checking además mira "¿le dejé opciones válidas
#     a los que vienen después?".
#
# Trazas:
#   Imprimimos:
#     • la variable que se asigna,
#     • el valor intentado,
#     • cómo se podan dominios vecinos,
#     • cuándo hacemos backtrack y restauramos dominios.
#
# =========================================================

from typing import Dict, List, Optional
import copy

# ---------------------------------------------------------
# 1. Definición del CSP (misma base que 17 y 18)
# ---------------------------------------------------------

variables: List[str] = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']

dominio_inicial: Dict[str, List[str]] = {
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
    """
    True si NO hay conflicto entre var_a=val_a y var_b=val_b.
    Regla: si son vecinas, deben ser distintos colores.
    """
    if var_b not in adyacentes.get(var_a, []):
        return True
    return val_a != val_b


def es_consistente_parcial(asignacion: Dict[str, str],
                           var: str,
                           valor: str) -> bool:
    """
    Verifica si puedo asignar var=valor sin violar restricciones
    con las variables YA asignadas.
    """
    for otra_var, otro_val in asignacion.items():
        if not no_conflicto_binario(var, valor, otra_var, otro_val):
            return False
    return True


# ---------------------------------------------------------
# 2. Forward Checking (poda de dominios futuros)
# ---------------------------------------------------------
def podar_dominios(dominios: Dict[str, List[str]],
                   asignacion: Dict[str, str],
                   var_actual: str,
                   valor_actual: str,
                   nivel: int) -> Optional[Dict[str, List[str]]]:
    """
    Aplica comprobación hacia delante al asignar var_actual = valor_actual.

    Para cada vecino NO asignado:
      - elimina del dominio cualquier valor que entre en conflicto.
      - si algún dominio queda vacío => return None (fracaso inmediato).

    Regresa una copia nueva de dominios podados SI ES VÁLIDA,
    o None si detectamos inconsistencia.
    """
    nuevos_dominios = copy.deepcopy(dominios)

    # Asegurar que var_actual queda fija a [valor_actual]
    nuevos_dominios[var_actual] = [valor_actual]

    for vecino in adyacentes[var_actual]:
        # solo interesa si el vecino NO está ya asignado
        if vecino in asignacion:
            continue

        dominio_vecino = nuevos_dominios[vecino]
        dominio_filtrado = []

        for val_vecino in dominio_vecino:
            # mantenemos solo valores que NO chocan con var_actual=valor_actual
            if no_conflicto_binario(var_actual, valor_actual, vecino, val_vecino):
                dominio_filtrado.append(val_vecino)

        print(" " * (2 * nivel) + f"    -> dominio[{vecino}] antes: {dominio_vecino}  después: {dominio_filtrado}")

        # Si el dominio del vecino queda vacío, fallo inmediato
        if len(dominio_filtrado) == 0:
            print(" " * (2 * nivel) + f"    !! dominio[{vecino}] quedó vacío -> inconsistencia temprana")
            return None

        nuevos_dominios[vecino] = dominio_filtrado

    return nuevos_dominios


# ---------------------------------------------------------
# 3. Backtracking con Forward Checking
# ---------------------------------------------------------
def forward_checking(asignacion_parcial: Dict[str, str],
                     orden_vars: List[str],
                     dominios: Dict[str, List[str]],
                     nivel: int = 0) -> Optional[Dict[str, str]]:
    """
    Versión de backtracking que hace comprobación hacia delante.
    - asignacion_parcial: variables ya fijadas
    - orden_vars: orden en que intentamos asignar
    - dominios: dominios actuales (ya podados hasta ahora)
    """

    # Caso base: ¿ya asignamos todo?
    if len(asignacion_parcial) == len(orden_vars):
        print(" " * (2 * nivel) + f">> ÉXITO: asignación completa")
        print(" " * (2 * nivel) + f"   {asignacion_parcial}\n")
        return asignacion_parcial

    # Elegimos la siguiente variable sin asignar
    var_actual = orden_vars[len(asignacion_parcial)]

    print(" " * (2 * nivel) + f"- Asignando {var_actual}")
    print(" " * (2 * nivel) + f"  Dominio disponible de {var_actual}: {dominios[var_actual]}")

    # Intentar cada valor posible en el dominio actual de la variable
    for valor in dominios[var_actual]:
        print(" " * (2 * nivel) + f"  Probar {var_actual} = {valor} ...", end=" ")

        # Checamos consistencia local con lo ya asignado
        if not es_consistente_parcial(asignacion_parcial, var_actual, valor):
            print("CONFLICTO LOCAL")
            continue

        print("OK (consistente). Ahora aplico forward checking:")
        # Asignar temporalmente
        nueva_asignacion = dict(asignacion_parcial)
        nueva_asignacion[var_actual] = valor

        # Hacer poda de dominios futuros
        dominios_podados = podar_dominios(dominios,
                                          nueva_asignacion,
                                          var_actual,
                                          valor,
                                          nivel)
        if dominios_podados is None:
            print(" " * (2 * nivel) + f"    -> Inconsistencia detectada. Backtrack inmediato.\n")
            continue

        # Llamada recursiva con asignación extendida y dominios reducidos
        resultado = forward_checking(nueva_asignacion,
                                     orden_vars,
                                     dominios_podados,
                                     nivel + 1)

        if resultado is not None:
            return resultado  # éxito propagado

        print(" " * (2 * nivel) + f"  (Backtrack) {var_actual} = {valor} no lleva a solución.\n")

    # Si ningún valor funcionó, fallamos aquí
    print(" " * (2 * nivel) + f"!! Fallo en {var_actual}, regreso nivel arriba\n")
    return None


# ---------------------------------------------------------
# 4. MAIN: Ejecutar Forward Checking con trazas
# ---------------------------------------------------------
if __name__ == "__main__":
    print("==============================================")
    print("TRACE Forward Checking (Comprobación Hacia Delante)")
    print("==============================================\n")

    orden = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']

    solucion = forward_checking(
        asignacion_parcial={},
        orden_vars=orden,
        dominios=copy.deepcopy(dominio_inicial)
    )

    print("==============================================")
    print("RESULTADO FINAL FORWARD CHECKING")
    print("==============================================")
    print("Asignación encontrada:", solucion)
    print("¿Solución completa?  :", solucion is not None and len(solucion) == len(variables))
    print("")

    # Nota:
    # Forward Checking mejora respecto al backtracking básico
    # porque "mira hacia adelante" y elimina valores que ya no
    # son posibles en las variables aún no asignadas.
    #
    # Esto evita explorar ramas que ya se sabe que van a fallar.
    # En el módulo 20 vamos a empujar esto todavía más lejos:
    # Propagación de Restricciones (consistencia de arcos),
    # donde no solo reducimos dominios de los vecinos directos,
    # sino que propagamos esas reducciones más allá.
