# =========================================================
# 11 - ELIMINACIÓN DE VARIABLES
# ---------------------------------------------------------
# Descripción:
#   En este script implementamos la idea de
#   Eliminación de Variables (Variable Elimination),
#   que es una optimización de la inferencia por enumeración.
#
#   Objetivo:
#     Calcular P(Query | Evidencia)
#     sin tener que enumerar TODAS las combinaciones completas
#     de todas las variables ocultas de la red.
#
#   ¿Qué hace?
#     En lugar de expandir toda la distribución conjunta
#     y luego sumar al final,
#     vamos "eliminando" (sumando) variables ocultas una por una,
#     combinando factores locales.
#
#   Intuición:
#     - Piensa en las probabilidades como "factores".
#     - Multiplicas factores que contienen cierta variable oculta.
#     - Luego haces suma marginal sobre esa variable para
#       "eliminarla".
#     - Esto produce un nuevo factor que ya no depende
#       de esa variable.
#
#   En redes pequeñas se parece mucho a enumeración;
#   en redes grandes ahorra muchísimo.
#
# Ejemplo visual (misma red):
#
#     Lluvia  ──►  SensorAgua  ──►  Alarma
#
#   Consulta:
#     P(Lluvia | Alarma=True)
#
#   Variable oculta a eliminar:
#     SensorAgua
#
#   Esquema:
#     1. Partimos de factores:
#        f1(Lluvia) = P(Lluvia)
#        f2(SensorAgua, Lluvia) = P(SensorAgua | Lluvia)
#        f3(Alarma, SensorAgua) = P(Alarma | SensorAgua)
#
#     2. Fijamos evidencia Alarma=True en f3.
#
#     3. Multiplicamos f2 y f3 (comparten SensorAgua),
#        luego sumamos sobre SensorAgua para eliminarla.
#
#     4. Multiplicamos resultado por f1(Lluvia).
#
#     5. Normalizamos para obtener P(Lluvia | Alarma=True).
#
# =========================================================

from typing import Dict, Tuple, List
import itertools

# ---------------------------------------------------------
# 1. Tablas de probabilidad de la red bayesiana
# ---------------------------------------------------------

# P(Lluvia)
P_Lluvia: Dict[bool, float] = {
    True: 0.3,
    False: 0.7
}

# P(SensorAgua | Lluvia)
# clave: (SensorAgua, Lluvia)
P_SensorAgua_dado_Lluvia: Dict[Tuple[bool, bool], float] = {
    (True,  True): 0.9,
    (False, True): 0.1,
    (True,  False): 0.2,
    (False, False): 0.8
}

# P(Alarma | SensorAgua)
# clave: (Alarma, SensorAgua)
P_Alarma_dado_Sensor: Dict[Tuple[bool, bool], float] = {
    (True,  True): 0.95,
    (False, True): 0.05,
    (True,  False): 0.01,
    (False, False): 0.99
}

# ---------------------------------------------------------
# 2. Representar factores
# ---------------------------------------------------------
# Un factor lo representaremos como:
# {
#   tupla_de_asignacion: valor_prob
# }
#
# donde tupla_de_asignacion es un dict "parcheado" en forma de tuple ordenada.
#
# Pero para mantenerlo muy claro y didáctico,
# vamos a representar factores como listas de filas legibles.
#
# Cada factor tendrá:
#   - vars: lista del nombre de las variables en orden
#   - tabla: lista de dicts con {'asign': {...}, 'p': ...}
#
# Ejemplo de factor f2(SensorAgua,Lluvia):
#   vars = ["SensorAgua","Lluvia"]
#   tabla = [
#      {'asign': {'SensorAgua': True,  'Lluvia': True },  'p': 0.9},
#      {'asign': {'SensorAgua': False, 'Lluvia': True },  'p': 0.1},
#      {'asign': {'SensorAgua': True,  'Lluvia': False},  'p': 0.2},
#      {'asign': {'SensorAgua': False, 'Lluvia': False},  'p': 0.8},
#   ]
#
# Esto nos permitirá multiplicar factores y hacer sumas marginales paso a paso.

def factor_P_Lluvia():
    f = {
        "vars": ["Lluvia"],
        "tabla": [
            {"asign": {"Lluvia": True},  "p": P_Lluvia[True]},
            {"asign": {"Lluvia": False}, "p": P_Lluvia[False]},
        ]
    }
    return f

def factor_P_SensorAgua_given_Lluvia():
    f = {
        "vars": ["SensorAgua","Lluvia"],
        "tabla": [
            {"asign": {"SensorAgua": True,  "Lluvia": True },  "p": 0.9},
            {"asign": {"SensorAgua": False, "Lluvia": True },  "p": 0.1},
            {"asign": {"SensorAgua": True,  "Lluvia": False},  "p": 0.2},
            {"asign": {"SensorAgua": False, "Lluvia": False},  "p": 0.8},
        ]
    }
    return f

def factor_P_Alarma_given_SensorAgua():
    f = {
        "vars": ["Alarma","SensorAgua"],
        "tabla": [
            {"asign": {"Alarma": True,  "SensorAgua": True },  "p": 0.95},
            {"asign": {"Alarma": False, "SensorAgua": True },  "p": 0.05},
            {"asign": {"Alarma": True,  "SensorAgua": False},  "p": 0.01},
            {"asign": {"Alarma": False, "SensorAgua": False},  "p": 0.99},
        ]
    }
    return f

# ---------------------------------------------------------
# 3. Restringir un factor con evidencia
# ---------------------------------------------------------
# Si tenemos evidencia, ej. Alarma=True,
# entonces en el factor P(Alarma|SensorAgua)
# sólo nos quedamos con las filas que cumplan Alarma=True.
#
def restringir_factor(factor, evidencia: Dict[str,bool]):
    """
    Devuelve un factor NUEVO donde forzamos la evidencia.
    Eliminamos las filas que no coinciden.
    También podemos quitar la variable de evidencia de la lista de vars.
    """
    nuevas_vars = [v for v in factor["vars"] if v not in evidencia]
    nueva_tabla = []

    for fila in factor["tabla"]:
        asign_ok = True
        for var_ev, val_ev in evidencia.items():
            if var_ev in fila["asign"] and fila["asign"][var_ev] != val_ev:
                asign_ok = False
                break
        if asign_ok:
            # Construimos una fila reducida (quitando variables fijadas)
            nueva_asign = {
                var: fila["asign"][var]
                for var in fila["asign"]
                if var not in evidencia
            }
            nueva_tabla.append({"asign": nueva_asign, "p": fila["p"]})

    return {"vars": nuevas_vars, "tabla": nueva_tabla}

# ---------------------------------------------------------
# 4. Multiplicar factores
# ---------------------------------------------------------
# Para multiplicar factores f(X,Y) y g(Y,Z):
#   - El nuevo factor tendrá vars = X ∪ Y ∪ Z
#   - Para cada combinación que sea consistente en las vars comunes,
#     multiplicamos las probabilidades.
#
def multiplicar_factores(f1, f2):
    nuevas_vars = list(dict.fromkeys(f1["vars"] + f2["vars"]))  # union preservando orden
    nueva_tabla = []

    for fila1 in f1["tabla"]:
        for fila2 in f2["tabla"]:
            # Checar consistencia en las variables en común
            consistente = True
            for v in f1["vars"]:
                if v in f2["vars"]:
                    if fila1["asign"][v] != fila2["asign"][v]:
                        consistente = False
                        break
            if not consistente:
                continue

            # Mezclar asignaciones
            asign_merged = dict(fila1["asign"])
            asign_merged.update(fila2["asign"])

            nueva_tabla.append({
                "asign": asign_merged,
                "p": fila1["p"] * fila2["p"]
            })

    return {"vars": nuevas_vars, "tabla": nueva_tabla}

# ---------------------------------------------------------
# 5. Sumar (marginalizar) una variable en un factor
# ---------------------------------------------------------
# Ejemplo:
#   tenemos un factor f(SensorAgua, Lluvia)
#   queremos "eliminar" SensorAgua
#
#   Resultado será un factor sólo sobre {Lluvia}
#   donde para cada valor de Lluvia sumamos
#   las probabilidades sobre SensorAgua=True y False.
#
def sumar_sobre_variable(factor, var_a_eliminar: str):
    nuevas_vars = [v for v in factor["vars"] if v != var_a_eliminar]

    # Para agrupar filas que sólo difieren en var_a_eliminar,
    # usamos un diccionario clave -> suma
    # clave = tupla ordenada con las vars que quedan
    acumulador = {}

    for fila in factor["tabla"]:
        # clave basada en las vars restantes
        clave = tuple((v, fila["asign"][v]) for v in nuevas_vars)
        acumulador.setdefault(clave, 0.0)
        acumulador[clave] += fila["p"]

    # Convertimos acumulador de regreso a formato de factor
    nueva_tabla = []
    for clave, ptotal in acumulador.items():
        asign_dict = {var: val for (var,val) in clave}
        nueva_tabla.append({"asign": asign_dict, "p": ptotal})

    return {"vars": nuevas_vars, "tabla": nueva_tabla}


# ---------------------------------------------------------
# 6. Procedimiento específico para P(Lluvia | Alarma=True)
# ---------------------------------------------------------
# Paso a paso:
#   factores iniciales:
#       f1(Lluvia)
#       f2(SensorAgua, Lluvia)
#       f3(Alarma, SensorAgua)
#
#   1) Fijamos evidencia Alarma=True en f3  -> f3e(SensorAgua)
#   2) Multiplicamos f2 y f3e -> f23(Lluvia, SensorAgua)
#   3) Sumamos/eliminamos SensorAgua de f23 -> f23sum(Lluvia)
#   4) Multiplicamos f1(Lluvia) * f23sum(Lluvia) -> f_final(Lluvia)
#   5) Normalizamos para obtener P(Lluvia | Alarma=True)

def inferencia_por_eliminacion():
    # 1) Crear factores base
    f1 = factor_P_Lluvia()                  # vars: [Lluvia]
    f2 = factor_P_SensorAgua_given_Lluvia() # vars: [SensorAgua,Lluvia]
    f3 = factor_P_Alarma_given_SensorAgua() # vars: [Alarma,SensorAgua]

    print("==============================================")
    print("TRACE ELIMINACIÓN DE VARIABLES")
    print("==============================================\n")

    print("1) Factores iniciales:")
    print("   f1(Lluvia) = P(Lluvia)")
    for fila in f1["tabla"]:
        print(f"     {fila['asign']} -> {fila['p']:.3f}")
    print("")
    print("   f2(SensorAgua,Lluvia) = P(SensorAgua | Lluvia)")
    for fila in f2["tabla"]:
        print(f"     {fila['asign']} -> {fila['p']:.3f}")
    print("")
    print("   f3(Alarma,SensorAgua) = P(Alarma | SensorAgua)")
    for fila in f3["tabla"]:
        print(f"     {fila['asign']} -> {fila['p']:.3f}")
    print("")

    # 2) Evidencia: Alarma=True  -> restringimos f3
    evidencia = {"Alarma": True}
    f3e = restringir_factor(f3, evidencia)

    print("2) Restringimos evidencia Alarma=True en f3:")
    print("   f3e(SensorAgua) ahora es:")
    for fila in f3e["tabla"]:
        print(f"     {fila['asign']} -> {fila['p']:.3f}")
    print("")

    # 3) Multiplicamos f2 y f3e
    f23 = multiplicar_factores(f2, f3e)

    print("3) Multiplicamos f2(SensorAgua,Lluvia) * f3e(SensorAgua):")
    print("   f23(Lluvia,SensorAgua):")
    for fila in f23["tabla"]:
        print(f"     {fila['asign']} -> {fila['p']:.6f}")
    print("")

    # 4) Eliminamos SensorAgua sumándolo
    f23sum = sumar_sobre_variable(f23, "SensorAgua")

    print("4) Sumamos sobre SensorAgua para eliminarlo:")
    print("   f23sum(Lluvia):")
    for fila in f23sum["tabla"]:
        print(f"     {fila['asign']} -> {fila['p']:.6f}")
    print("")

    # 5) Multiplicamos con f1(Lluvia)
    f_final = multiplicar_factores(f1, f23sum)

    print("5) Multiplicamos con f1(Lluvia) = P(Lluvia):")
    print("   f_final(Lluvia) [sin normalizar]:")
    for fila in f_final["tabla"]:
        print(f"     {fila['asign']} -> {fila['p']:.6f}")
    print("")

    # 6) Normalizamos para que sea distribución sobre Lluvia
    suma_total = sum(f["p"] for f in f_final["tabla"])
    distrib = {}
    for fila in f_final["tabla"]:
        val_lluvia = fila["asign"]["Lluvia"]
        distrib[val_lluvia] = fila["p"] / suma_total

    print("6) Normalizamos para obtener P(Lluvia | Alarma=True):")
    print(f"   P(Lluvia=True  | Alarma=True)  = {distrib[True]:.4f}")
    print(f"   P(Lluvia=False | Alarma=True)  = {distrib[False]:.4f}\n")

    print("INTERPRETACIÓN:")
    print("- Eliminación de variables hace lo MISMO que la enumeración,")
    print("  pero sin volver a recalcular las mismas combinaciones mil veces.")
    print("")
    print("- Aquí sólo teníamos una variable oculta (SensorAgua),")
    print("  así que la ganancia no es enorme.")
    print("  Pero en redes grandes, este método es MUCHO más eficiente.")
    print("")

if __name__ == "__main__":
    inferencia_por_eliminacion()

# Nota:
# - En los siguientes scripts:
#     12) Muestreo Directo / Por Rechazo
#     13) Ponderación de Verosimilitud
#     14) Monte Carlo para Cadenas de Markov
#
#   ya no haremos inferencia exacta, sino APROXIMADA.
#   Ahí vamos a generar muestras aleatorias del mundo
#   y estimar probabilidades empíricamente.
#
# - En esos métodos sí puede ser útil graficar histogramas
#   de frecuencias simuladas vs probabilidades teóricas.
#   Te voy a avisar antes de meter matplotlib para graficar.
