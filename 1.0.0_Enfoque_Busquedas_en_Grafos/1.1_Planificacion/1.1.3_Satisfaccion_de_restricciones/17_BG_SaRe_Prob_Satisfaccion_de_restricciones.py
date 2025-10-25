# =========================================================
# 17 - PROBLEMAS DE SATISFACCIÓN DE RESTRICCIONES (CSP Base)
# ---------------------------------------------------------
# Descripción:
#   Este módulo define la estructura base de un problema de
#   Satisfacción de Restricciones (CSP - Constraint Satisfaction Problem).
#
#   El CSP modela el clásico problema de "coloreado de mapa":
#   asignar un color a cada región del mapa de Australia de forma
#   que ninguna región vecina comparta el mismo color.
#
#   Este problema se usa ampliamente en Inteligencia Artificial
#   para ilustrar cómo los algoritmos pueden encontrar soluciones
#   compatibles bajo restricciones, sin necesidad de explorar
#   todas las combinaciones posibles.
#
#   Lo importante es entender que:
#     - Cada región es una variable.
#     - Los colores posibles son su dominio.
#     - Las relaciones de vecindad definen las restricciones.
#
# Características del modelo:
#   • Variables: regiones ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
#   • Dominio: {'ROJO', 'VERDE', 'AZUL'}
#   • Restricciones binarias: regiones vecinas ≠ mismo color
#
# Contexto:
#   Este mismo CSP será reutilizado en los siguientes algoritmos:
#     18 - Búsqueda de Vuelta Atrás
#     19 - Comprobación Hacia Delante
#     20 - Propagación de Restricciones
#     21 - Salto Atrás Dirigido por Conflictos
#     22 - Búsqueda Local: Mínimos-Conflictos
#     23 - Acondicionamiento del Corte
#
# =========================================================

from typing import Dict, List, Set, Optional

# ---------------------------------------------------------
# 1. Definición del CSP
# ---------------------------------------------------------

# Variables del problema (regiones)
variables: List[str] = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']

# Dominio posible para cada variable
dominio_base: Dict[str, List[str]] = {
    var: ['ROJO', 'VERDE', 'AZUL'] for var in variables
}

# Adyacencias (quién es vecino de quién)
# Si A y B son vecinos: A != B
adyacentes: Dict[str, List[str]] = {
    'WA':  ['NT', 'SA'],
    'NT':  ['WA', 'SA', 'Q'],
    'SA':  ['WA', 'NT', 'Q', 'NSW', 'V'],
    'Q':   ['NT', 'SA', 'NSW'],
    'NSW': ['SA', 'Q', 'V'],
    'V':   ['SA', 'NSW', 'T'],
    'T':   ['V']
}


# ---------------------------------------------------------
# 2. Funciones de validación de restricciones
# ---------------------------------------------------------
def no_conflicto_binario(var_a: str, val_a: str,
                         var_b: str, val_b: str) -> bool:
    """
    Revisa la restricción binaria entre (var_a, val_a) y (var_b, val_b).

    Regla:
    - Si var_a y var_b son vecinos, entonces val_a != val_b.
    - Si NO son vecinos, no hay restricción entre ellas.

    Regresa True si NO hay conflicto.
    Regresa False si HAY conflicto.
    """
    # Si no son vecinos, no hay restricción que violar
    if var_b not in adyacentes.get(var_a, []):
        return True

    # Son vecinos -> no pueden tener el mismo color
    return val_a != val_b


def es_asignacion_valida_parcial(asignacion: Dict[str, str]) -> bool:
    """
    Verifica si una asignación PARCIAL respeta todas las restricciones
    entre las variables YA asignadas.

    Ejemplo:
      asignacion = { 'WA':'ROJO', 'NT':'ROJO' }
      -> esto es inválido porque WA y NT son vecinas y tienen el mismo color

    Nota:
    - Solo se revisan pares de variables que YA estén asignadas.
    - No exige que todas las variables estén asignadas.
    """
    vars_asignadas = list(asignacion.keys())
    for i in range(len(vars_asignadas)):
        for j in range(i + 1, len(vars_asignadas)):
            vi = vars_asignadas[i]
            vj = vars_asignadas[j]
            vali = asignacion[vi]
            valj = asignacion[vj]

            if not no_conflicto_binario(vi, vali, vj, valj):
                return False
    return True


def es_asignacion_completa_valida(asignacion: Dict[str, str]) -> bool:
    """
    Verifica si TODAS las variables están asignadas y
    si TODAS las restricciones se cumplen.

    Es básicamente es_asignacion_valida_parcial(...)
    + checar que no falte ninguna variable.
    """
    # ¿faltan variables sin asignar?
    if set(asignacion.keys()) != set(variables):
        return False

    # ¿rompe alguna restricción binaria?
    return es_asignacion_valida_parcial(asignacion)


# ---------------------------------------------------------
# 3. Impresión / traza de demostración
# ---------------------------------------------------------
if __name__ == "__main__":
    print("==============================================")
    print("TRACE CSP BASE")
    print("==============================================\n")

    print("Variables:", variables)
    print("Dominio común:", dominio_base['WA'])
    print("Restricciones de vecindad (A != B):")
    for reg, vecinos in adyacentes.items():
        print(f"  {reg} -> {vecinos}")
    print("")

    # Ejemplo 1: Asignación parcial válida
    asignacion_parcial_1 = {
        'WA': 'ROJO',
        'NT': 'VERDE',
        'SA': 'AZUL'
    }
    print("Probando asignación parcial 1:", asignacion_parcial_1)
    print("¿Parcial válida?:", es_asignacion_valida_parcial(asignacion_parcial_1))
    print("¿Completa válida?:", es_asignacion_completa_valida(asignacion_parcial_1))
    print("")

    # Ejemplo 2: Asignación parcial con conflicto
    asignacion_parcial_2 = {
        'WA': 'ROJO',
        'NT': 'ROJO'   # <- CONFLICTO: WA y NT son vecinas
    }
    print("Probando asignación parcial 2:", asignacion_parcial_2)
    print("¿Parcial válida?:", es_asignacion_valida_parcial(asignacion_parcial_2))
    print("")

    # Ejemplo 3: Asignación completa válida hipotética (solo ejemplo)
    asignacion_completa = {
        'WA':  'ROJO',
        'NT':  'VERDE',
        'SA':  'AZUL',
        'Q':   'ROJO',
        'NSW': 'VERDE',
        'V':   'ROJO',
        'T':   'AZUL'
    }
    print("Probando asignación completa:", asignacion_completa)
    print("¿Completa válida?:", es_asignacion_completa_valida(asignacion_completa))
    print("")

    # Nota:
    # Este módulo define:
    #  - variables
    #  - dominios
    #  - restricciones
    #  - funciones de consistencia
    #
    # Es la base para los siguientes algoritmos (18–23)
    # que implementarán distintos métodos de búsqueda y
    # verificación sobre este mismo problema.
