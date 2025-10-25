# =========================================================
# 23 - ACONDICIONAMIENTO DEL CORTE (Cutset Conditioning)
# ---------------------------------------------------------
# Descripción:
#   Este módulo implementa una versión simplificada del
#   algoritmo de **Acondicionamiento del Corte (Cutset Conditioning)**,
#   una técnica avanzada para resolver CSP mediante
#   **reducción estructural** del problema.
#
#   En lugar de hacer solo poda o búsqueda local,
#   esta técnica se enfoca en **romper ciclos del grafo
#   de restricciones** para convertirlo en un árbol (un
#   problema acíclico), lo cual se puede resolver más
#   fácilmente mediante propagación local.
#
#   Idea central:
#     1. Identificar un conjunto de variables (cutset)
#        cuya eliminación (fijación) vuelve acíclico el grafo.
#     2. Enumerar todas las combinaciones posibles de valores
#        para esas variables del cutset.
#     3. Para cada combinación:
#           - fijar esas variables,
#           - resolver el resto del CSP (ahora sin ciclos)
#             mediante propagación o backtracking simple.
#     4. Si alguna combinación completa el CSP sin conflictos,
#        devolvemos esa solución.
#
#   Esto evita reexplorar ramas redundantes y explota
#   la estructura del problema.
#
# Modelo CSP:
#   - Variables: ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
#   - Dominio: { 'ROJO', 'VERDE', 'AZUL' }
#   - Restricciones binarias: regiones vecinas ≠ mismo color
#
#   (Coloreo del mapa de Australia)
#
# Trazas:
#   - Mostramos el conjunto de corte elegido.
#   - Probamos todas las combinaciones de valores para él.
#   - Verificamos si al fijarlas se puede resolver el resto.
#
# =========================================================

from typing import Dict, List, Tuple, Optional
import itertools

# ---------------------------------------------------------
# 1. Definición del CSP (misma base)
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

def no_conflicto_binario(var_a: str, val_a: str,
                         var_b: str, val_b: str) -> bool:
    """True si var_a=val_a y var_b=val_b NO violan la restricción."""
    if var_b not in adyacentes.get(var_a, []):
        return True
    return val_a != val_b


def es_consistente(asignacion: Dict[str, str],
                   var: str,
                   valor: str) -> bool:
    """Revisa consistencia con la asignación parcial."""
    for otra_var, otro_val in asignacion.items():
        if not no_conflicto_binario(var, valor, otra_var, otro_val):
            return False
    return True


# ---------------------------------------------------------
# 2. Backtracking simple (para el subproblema sin ciclos)
# ---------------------------------------------------------
def backtracking_simple(asignacion: Dict[str, str],
                        vars_restantes: List[str]) -> Optional[Dict[str, str]]:
    """Backtracking básico sin poda, usado para resolver el subproblema acíclico."""
    if not vars_restantes:
        return asignacion

    var = vars_restantes[0]

    for valor in dominio[var]:
        if es_consistente(asignacion, var, valor):
            nueva_asig = dict(asignacion)
            nueva_asig[var] = valor
            resultado = backtracking_simple(nueva_asig, vars_restantes[1:])
            if resultado:
                return resultado

    return None


# ---------------------------------------------------------
# 3. Cutset Conditioning
# ---------------------------------------------------------
def cutset_conditioning(
    cutset: List[str]
) -> Optional[Dict[str, str]]:
    """
    Implementación didáctica de Acondicionamiento del Corte.

    Recibe un conjunto de variables 'cutset' que al fijarse rompen
    los ciclos del CSP. Probaremos todas las combinaciones posibles
    de sus valores y resolveremos el resto con backtracking simple.
    """

    print("==============================================")
    print("TRACE Acondicionamiento del Corte (Cutset Conditioning)")
    print("==============================================\n")

    print(f"Conjunto de corte elegido: {cutset}\n")

    # Variables restantes (el subproblema acíclico)
    restantes = [v for v in variables if v not in cutset]

    # Enumeramos todas las combinaciones posibles para el cutset
    dominios_cutset = [dominio[v] for v in cutset]
    combinaciones = list(itertools.product(*dominios_cutset))

    for idx, combinacion in enumerate(combinaciones):
        print(f"[Prueba {idx+1}] Fijando corte {dict(zip(cutset, combinacion))}")

        asignacion_inicial = dict(zip(cutset, combinacion))

        # Verificar que la asignación inicial del cutset no tenga conflictos internos
        consistente = True
        for i in range(len(cutset)):
            for j in range(i+1, len(cutset)):
                vi, vj = cutset[i], cutset[j]
                if not no_conflicto_binario(vi, asignacion_inicial[vi], vj, asignacion_inicial[vj]):
                    consistente = False
        if not consistente:
            print("   ✗ El corte tiene conflictos internos, se descarta.\n")
            continue

        print("   ✓ Corte consistente. Resolviendo subproblema restante...\n")

        resultado = backtracking_simple(asignacion_inicial, restantes)
        if resultado:
            print(f"   ✅ Solución encontrada con esta combinación: {resultado}\n")
            return resultado
        else:
            print("   ✗ No se logró solución con este corte.\n")

    print("==============================================")
    print("No se encontró solución en ninguna combinación del corte.")
    print("==============================================\n")
    return None


# ---------------------------------------------------------
# 4. MAIN: Ejecutar Cutset Conditioning
# ---------------------------------------------------------
if __name__ == "__main__":
    # En el mapa de Australia, podemos romper los ciclos
    # si fijamos una variable clave como SA, que conecta
    # con casi todas las demás (es el "hub" del grafo).
    #
    # Al eliminar SA, el resto del grafo se vuelve casi acíclico.
    #
    # Por tanto, elegimos cutset = ['SA']

    cutset = ['SA']

    solucion = cutset_conditioning(cutset)

    print("==============================================")
    print("RESULTADO FINAL ACONDICIONAMIENTO DEL CORTE")
    print("==============================================")
    print("Asignación encontrada:", solucion)
    print("¿Solución completa?:", solucion is not None and len(solucion) == len(variables))
    print("")

    # Nota:
    # - Acondicionamiento del Corte reduce un CSP con ciclos
    #   a varios CSP acíclicos más fáciles de resolver.
    #
    # - Es una técnica híbrida entre análisis estructural
    #   (grafo de restricciones) y búsqueda.
    #
    # - En CSP grandes se usa junto con heurísticas de
    #   ordenamiento y detección automática de cutsets
    #   mínimos para reducir el espacio de búsqueda drásticamente.
