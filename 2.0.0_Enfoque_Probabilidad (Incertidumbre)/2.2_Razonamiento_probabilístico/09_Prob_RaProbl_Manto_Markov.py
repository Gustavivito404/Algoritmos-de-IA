# =========================================================
# 09 - MANTO DE MARKOV
# ---------------------------------------------------------
# Descripción:
#   En este script explicamos y calculamos el
#   MANTO DE MARKOV (Markov Blanket) de una variable
#   dentro de una Red Bayesiana.
#
#   El Manto de Markov de un nodo X es el conjunto mínimo
#   de variables que "protege" (aísla) a X del resto de la red.
#
#   Intuición:
#     Si conoces el Manto de Markov de X,
#     entonces X es independiente del resto del universo.
#
#   Formalmente:
#     El manto de Markov de un nodo X incluye:
#       - Sus PADRES
#       - Sus HIJOS
#       - Los OTROS PADRES de sus hijos (co-padres)
#
# Características:
#   - Te dice qué información local necesitas para
#     razonar sobre una variable sin mirar toda la red.
#   - Es esencial en:
#       • Inferencia Bayesiana Local
#       • Muestreo Gibbs / MCMC
#       • Aprendizaje estructural en redes bayesianas
#
# Ejemplo visual (misma red que en 07 y 08):
#
#     Lluvia  ──►  SensorAgua  ──►  Alarma
#
#   Padres(SensorAgua)      = {Lluvia}
#   Hijos(SensorAgua)       = {Alarma}
#   Co-padres(SensorAgua)   = (otros padres de Alarma además de SensorAgua)
#                             En este caso no hay otros, entonces ∅.
#
#   Entonces:
#      MantoDeMarkov(SensorAgua) = { Lluvia, Alarma }
#
#   Nota:
#      Para "SensorAgua", saber Lluvia y Alarma hace que
#      el resto del mundo ya no aporte información adicional.
# =========================================================

from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------
# 1. Definimos la estructura de la red bayesiana
# ---------------------------------------------------------
# Usamos la red:
#
#   Lluvia  ->  SensorAgua  ->  Alarma
#
# Vamos a representarla explícitamente como:
#   - padres_de[nodo] = [lista de nodos padre]
#   - hijos_de[nodo]  = [lista de nodos hijo]

nodos = ["Lluvia", "SensorAgua", "Alarma"]

padres_de: Dict[str, List[str]] = {
    "Lluvia":      [],            # Lluvia no tiene padres
    "SensorAgua":  ["Lluvia"],    # SensorAgua depende de Lluvia
    "Alarma":      ["SensorAgua"] # Alarma depende de SensorAgua
}

hijos_de: Dict[str, List[str]] = {
    "Lluvia":      ["SensorAgua"],
    "SensorAgua":  ["Alarma"],
    "Alarma":      []
}

# ---------------------------------------------------------
# 2. Función para obtener el Manto de Markov de un nodo
# ---------------------------------------------------------
def manto_de_markov(nodo: str) -> Set[str]:
    """
    Construye el Manto de Markov del nodo dado.

    Manto(X) = padres(X)
               ∪ hijos(X)
               ∪ (padres de cada hijo de X)
    """
    resultado: Set[str] = set()

    # 1) Padres directos de X
    for p in padres_de[nodo]:
        resultado.add(p)

    # 2) Hijos directos de X
    for h in hijos_de[nodo]:
        resultado.add(h)

        # 3) Co-padres:
        #    Todos los padres de ese hijo, excepto X mismo,
        #    porque esos co-padres también influyen en ese hijo
        #    y por lo tanto nos dan información sobre X.
        for otro_padre in padres_de[h]:
            if otro_padre != nodo:
                resultado.add(otro_padre)

    return resultado

# ---------------------------------------------------------
# 3. Mostrar el Manto de Markov para cada variable
# ---------------------------------------------------------
print("==============================================")
print("TRACE MANTO DE MARKOV")
print("==============================================\n")

for variable in nodos:
    mm = manto_de_markov(variable)
    print(f"1) Variable objetivo: {variable}")
    print(f"   Padres({variable}) = {padres_de[variable]}")
    print(f"   Hijos({variable})  = {hijos_de[variable]}")

    # Recolectar co-padres explícitamente para mostrar
    copadres: Set[str] = set()
    for h in hijos_de[variable]:
        for p in padres_de[h]:
            if p != variable:
                copadres.add(p)

    print(f"   Co-padres({variable}) = {list(copadres)}")
    print(f"   => Manto de Markov({variable}) = {sorted(mm)}\n")

# ---------------------------------------------------------
# 4. Interpretación esencial
# ---------------------------------------------------------
print("2) Interpretación esencial:")
print("   El Manto de Markov(X) es TODO lo que necesito saber")
print("   para razonar sobre X, SIN mirar el resto de la red.\n")

print("   Ejemplo concreto con esta red:\n")

# Para 'SensorAgua'
mm_sensor = manto_de_markov("SensorAgua")
print("   Para SensorAgua:")
print(f"      MantoDeMarkov(SensorAgua) = {sorted(mm_sensor)}")
print("      = { 'Lluvia', 'Alarma' }")
print("")
print("   Significado en palabras:")
print("      Si ya conozco si llueve y si la alarma sonó,")
print("      entonces el resto del mundo no me da información extra")
print("      sobre el estado del sensor de agua.")
print("")

# Para 'Lluvia'
mm_lluvia = manto_de_markov("Lluvia")
print("   Para Lluvia:")
print(f"      MantoDeMarkov(Lluvia) = {sorted(mm_lluvia)}")
print("      = { 'SensorAgua' }")
print("")
print("   Significado en palabras:")
print("      Una vez que sé qué dice el sensor,")
print("      no necesito nada más de la red para razonar sobre Lluvia.\n")

# Para 'Alarma'
mm_alarma = manto_de_markov("Alarma")
print("   Para Alarma:")
print(f"      MantoDeMarkov(Alarma) = {sorted(mm_alarma)}")
print("      = { 'SensorAgua' }")
print("")
print("   Significado en palabras:")
print("      Lo único que directamente importa para explicar la alarma")
print("      (en esta red simple) es lo que reporta el sensor de agua.")
print("")

# ---------------------------------------------------------
# 5. Resumen final
# ---------------------------------------------------------
print("3) Resumen final:")
print("- El Manto de Markov de un nodo X incluye:")
print("    • Sus padres")
print("    • Sus hijos")
print("    • Los otros padres de sus hijos")
print("")
print("- Dato MUY importante en IA probabilística:")
print("    Saber el manto de X ⇒ X es independiente del resto")
print("    de la red, dado ese manto.")
print("")
print("- Esto permite hacer inferencia LOCAL,")
print("  muestrear variables individualmente (Gibbs sampling),")
print("  y reducir cálculo en modelos grandes.\n")

# Nota:
# - Este script no necesita gráfica porque aquí lo crítico
#   es la lista de quién influye directamente a quién.
# - Pero si quieres, podemos reutilizar el plot comentado
#   del script 07 para visualizar la red y así conectar
#   con lo que estás leyendo en las trazas.
