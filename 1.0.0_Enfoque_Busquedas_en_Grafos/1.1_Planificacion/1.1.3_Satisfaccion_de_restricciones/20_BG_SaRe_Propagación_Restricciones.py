# =========================================================
# 20 - PROPAGACIÓN DE RESTRICCIONES
# ---------------------------------------------------------
# Descripción:
#   En este módulo implementamos Propagación de Restricciones
#   mediante consistencia de arcos (tipo AC-3) aplicada dentro
#   de la búsqueda para CSP.
#
#   Objetivo:
#     Reducir dominios de TODAS las variables de manera
#     consistente antes/durante la búsqueda, no solo hacer
#     poda local como en Forward Checking.
#
#   Idea central (Consistencia de Arcos):
#     Para cada par (X, Y) que tiene restricción:
#        Para cada valor x en dominio[X],
#           debe existir AL MENOS un valor y en dominio[Y]
#           que NO viole la restricción entre X e Y.
#
#     Si NO existe tal y para cierto x,
#     entonces ese x se elimina del dominio de X.
#
#   Eso se repite y propaga,
#   porque al podar X puedo forzar podas en otros vecinos.
#
#   En este script vamos a:
#     1) Hacer búsqueda tipo backtracking.
#     2) Cada vez que asignamos una variable,
#        aplicamos:
#          - asignación fija
#          - forward checking básico
#          - y luego propagación AC-3 completa sobre los dominios.
#
# Diferencia con módulo 19 (Forward Checking):
#   - Forward Checking sólo mira vecinos directos de la variable recién asignada.
#   - Propagación de Restricciones hace un "efecto dominó":
#     si quitas valores de NT, eso puede obligar a quitar valores de Q,
#     que a su vez puede obligar a quitar valores de NSW, etc.
#
# Modelo CSP:
#   Variables:
#       ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
#   Dominio inicial:
#       { 'ROJO', 'VERDE', 'AZUL' }
#   Restricciones binarias:
#       regiones vecinas no pueden tener el mismo color
#
#   (Coloreo del mapa de Australia)
#
# Trazas:
#   - Cada asignación de variable.
#   - Podas de dominio.
#   - Pasos de AC-3 (cola de arcos a revisar).
#
# =========================================================

from typing import Dict, List, Tuple, Optional
import copy
from collections import deque

# ---------------------------------------------------------
# 1. Definición del CSP (misma base que 17-19)
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
    Regresa True si var_a=val_a y var_b=val_b NO violan la restricción.
    (Si son vecinas -> deben tener distinto color)
    """
    if var_b not in adyacentes.get(var_a, []):
        return True
    return val_a != val_b


def es_consistente_parcial(asignacion: Dict[str, str],
                           var: str,
                           valor: str) -> bool:
    """
    Checa si var=valor es consistente con lo que ya está asignado.
    """
    for otra_var, otro_val in asignacion.items():
        if not no_conflicto_binario(var, valor, otra_var, otro_val):
            return False
    return True


# ---------------------------------------------------------
# 2. AC-3: Consistencia de arcos
# ---------------------------------------------------------
def revisar_arco(xi: str, xj: str,
                 dominios: Dict[str, List[str]]) -> bool:
    """
    Revisa el arco (xi -> xj) e intenta podar el dominio de xi.
    Regresa True si el dominio de xi CAMBIÓ (podamos algo).
    Regresa False si no hubo cambios.

    Regla:
    Para cada valor v en dominio[xi],
      debe existir al menos UN valor w en dominio[xj]
      tal que no haya conflicto entre xi=v y xj=w.
    Si NO existe tal w, entonces v se elimina.

    Esto implementa la típica operación "Revise" de AC-3.
    """
    cambiado = False
    dominio_xi_actual = dominios[xi][:]  # copia para iterar

    for v in dominio_xi_actual:
        # ¿Existe algún w en dominio[xj] que sea compatible?
        compatible = False
        for w in dominios[xj]:
            if no_conflicto_binario(xi, v, xj, w):
                compatible = True
                break

        # Si no hay ningún compatible, eliminar v de dominio[xi]
        if not compatible:
            dominios[xi].remove(v)
            cambiado = True

    return cambiado


def ac3(dominios: Dict[str, List[str]]) -> bool:
    """
    Aplica AC-3 sobre TODOS los arcos del CSP.
    Usa una cola de arcos (xi, xj). Mientras quitemos valores,
    debemos volver a revisar arcos que apuntan a xi.
    Si algún dominio queda vacío, devolvemos False (inconsistente).
    """
    cola = deque()

    # Inicializamos la cola con TODOS los arcos dirigidos (Xi, Xj)
    for xi in variables:
        for xj in adyacentes[xi]:
            cola.append((xi, xj))

    print("    [AC-3] Inicializando cola de arcos... total:", len(cola))

    while cola:
        xi, xj = cola.popleft()
        print(f"    [AC-3] Revisando arco ({xi} -> {xj})")
        antes = dominios[xi][:]

        if revisar_arco(xi, xj, dominios):
            despues = dominios[xi][:]
            print(f"      dominio[{xi}] podado: {antes} -> {despues}")

            # Si el dominio queda vacío = inconsistencia
            if len(dominios[xi]) == 0:
                print(f"      !! dominio[{xi}] vacío. AC-3 detecta inconsistencia global.")
                return False

            # Si cambiamos el dominio de xi,
            # debemos volver a checar los vecinos de xi (excepto xj)
            for xk in adyacentes[xi]:
                if xk != xj:
                    cola.append((xk, xi))

    return True


# ---------------------------------------------------------
# 3. Propagación de restricciones dentro de la búsqueda
#    (Backtracking + Forward Checking + AC-3)
# ---------------------------------------------------------
def podar_dominios_forward(
    dominios: Dict[str, List[str]],
    asignacion: Dict[str, str],
    var_actual: str,
    valor_actual: str,
    nivel: int
) -> Optional[Dict[str, List[str]]]:
    """
    Forward Checking local:
    Fija var_actual = valor_actual y restringe dominios
    de sus vecinos no asignados.
    Si algún vecino queda sin valores válidos, devolvemos None.
    """
    nuevos_dominios = copy.deepcopy(dominios)
    nuevos_dominios[var_actual] = [valor_actual]

    for vecino in adyacentes[var_actual]:
        if vecino in asignacion:
            continue

        before = nuevos_dominios[vecino][:]
        filtrado = [
            val for val in nuevos_dominios[vecino]
            if no_conflicto_binario(var_actual, valor_actual, vecino, val)
        ]

        print(" " * (2 * nivel) + f"    FC: dominio[{vecino}] {before} -> {filtrado}")

        if len(filtrado) == 0:
            print(" " * (2 * nivel) + f"    FC: dominio[{vecino}] quedó vacío -> inconsistencia temprana")
            return None

        nuevos_dominios[vecino] = filtrado

    return nuevos_dominios


def backtracking_con_propagacion(
    asignacion_parcial: Dict[str, str],
    orden_vars: List[str],
    dominios: Dict[str, List[str]],
    nivel: int = 0
) -> Optional[Dict[str, str]]:
    """
    Búsqueda en profundidad con:
      - Comprobación hacia delante (forward checking),
      - seguido de propagación AC-3 global en los dominios.
    """

    # Caso base: ¿ya asignamos todas las variables?
    if len(asignacion_parcial) == len(orden_vars):
        print(" " * (2 * nivel) + f">> ÉXITO: asignación completa")
        print(" " * (2 * nivel) + f"   {asignacion_parcial}\n")
        return asignacion_parcial

    # Siguiente variable por asignar
    var_actual = orden_vars[len(asignacion_parcial)]

    print(" " * (2 * nivel) + f"- Asignando {var_actual}")
    print(" " * (2 * nivel) + f"  Dominio[{var_actual}] actual:", dominios[var_actual])

    # Intentar cada valor en su dominio actual
    for valor in dominios[var_actual]:
        print(" " * (2 * nivel) + f"  Probar {var_actual} = {valor} ...", end=" ")

        # Checar consistencia local con asignación parcial
        if not es_consistente_parcial(asignacion_parcial, var_actual, valor):
            print("CONFLICTO LOCAL")
            continue

        print("OK (consistente). Aplicando forward checking y AC-3:")

        # Extender asignación
        nueva_asignacion = dict(asignacion_parcial)
        nueva_asignacion[var_actual] = valor

        # 1. Forward checking local sobre los vecinos
        dominios_fc = podar_dominios_forward(
            dominios,
            nueva_asignacion,
            var_actual,
            valor,
            nivel
        )
        if dominios_fc is None:
            print(" " * (2 * nivel) + f"    -> Inconsistencia en forward checking. Backtrack.\n")
            continue

        # 2. Propagación AC-3 global
        dominios_prop = copy.deepcopy(dominios_fc)
        print(" " * (2 * nivel) + f"    -> Llamando AC-3 para propagar restricciones...")
        if not ac3(dominios_prop):
            print(" " * (2 * nivel) + f"    -> AC-3 detecta inconsistencia. Backtrack.\n")
            continue

        # Llamada recursiva con dominios ya podados globalmente
        resultado = backtracking_con_propagacion(
            nueva_asignacion,
            orden_vars,
            dominios_prop,
            nivel + 1
        )

        if resultado is not None:
            return resultado  # éxito

        print(" " * (2 * nivel) + f"  (Backtrack) {var_actual} = {valor} no llevó a solución.\n")

    print(" " * (2 * nivel) + f"!! Fallo en {var_actual}, regreso nivel arriba\n")
    return None


# ---------------------------------------------------------
# 4. MAIN: Ejecutar búsqueda con Propagación de Restricciones
# ---------------------------------------------------------
if __name__ == "__main__":
    print("==============================================")
    print("TRACE Propagación de Restricciones (AC-3 dentro de la búsqueda)")
    print("==============================================\n")

    orden = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']

    solucion = backtracking_con_propagacion(
        asignacion_parcial={},
        orden_vars=orden,
        dominios=copy.deepcopy(dominio_inicial),
        nivel=0
    )

    print("==============================================")
    print("RESULTADO FINAL PROPAGACIÓN DE RESTRICCIONES")
    print("==============================================")
    print("Asignación encontrada:", solucion)
    print("¿Solución completa?  :", solucion is not None and len(solucion) == len(variables))
    print("")

    # Nota:
    # - Esta versión combina:
    #       backtracking
    #       + forward checking
    #       + AC-3 (propagación global)
    #
    # - Esto hace que detectemos inconsistencias mucho antes
    #   que el backtracking puro (18) y hasta más que el
    #   forward checking simple (19).
    #
    # - En el siguiente algoritmo (21),
    #   vamos a optimizar el retroceso:
    #   en vez de regresar siempre solo 1 nivel,
    #   haremos "salto atrás dirigido por conflictos"
    #   para brincar directo a la variable que causó el problema.
