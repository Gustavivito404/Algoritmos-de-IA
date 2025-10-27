# =========================================================
# 10 - INFERENCIA POR ENUMERACIÓN
# ---------------------------------------------------------
# Descripción:
#   En este script implementamos "Inferencia por Enumeración"
#   (Enumeration Inference) en una Red Bayesiana pequeña.
#
#   ¿Qué es esto?
#   - Queremos calcular una probabilidad como
#         P(Query | Evidencia)
#     por ejemplo:
#         P(Lluvia=True | Alarma=True)
#
#   - Para hacerlo de forma directa y exacta,
#     enumeramos (sumamos) sobre TODOS los valores posibles
#     de las variables ocultas.
#
#   Idea general:
#     P(Lluvia=True | Alarma=True)
#        = α * P(Lluvia=True , Alarma=True)
#        = α * Σ_{SensorAgua} P(Lluvia=True, SensorAgua, Alarma=True)
#
#     donde α es una constante de normalización
#     para que las probabilidades sumen 1.
#
# Características:
#   - Es CORRECTO y EXACTO.
#   - Pero escala MAL si hay muchas variables
#     (porque hay que enumerar todas las combinaciones).
#
#   Más adelante (11 - Eliminación de Variables)
#   vamos a optimizar esto.
#
# Ejemplo visual (misma red que antes):
#
#     Lluvia  ──►  SensorAgua  ──►  Alarma
#
#   Distribución conjunta:
#     P(L,S,A) = P(L) * P(S|L) * P(A|S)
#
#   Vamos a responder preguntas del tipo:
#     "Si sonó la alarma, ¿qué tan probable es que esté lloviendo?"
#
# =========================================================

from typing import Dict, Tuple, List
import itertools

# ---------------------------------------------------------
# 1. Red Bayesiana base (misma que en 07 y 08)
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

# Lista de variables y su orden causal (padres antes que hijos)
variables = ["Lluvia", "SensorAgua", "Alarma"]

# padres para cada variable (estructura del grafo)
padres_de = {
    "Lluvia": [],
    "SensorAgua": ["Lluvia"],
    "Alarma": ["SensorAgua"]
}

# ---------------------------------------------------------
# 2. Funciones auxiliares para evaluar factores P(X | padres)
# ---------------------------------------------------------
def prob_variable(nombre_var: str, valor: bool, asignacion: Dict[str, bool]) -> float:
    """
    Regresa P(var = valor | padres(var) = asignacion[padre])
    usando las tablas de nuestra red.
    """
    if nombre_var == "Lluvia":
        # Nodo raíz: P(Lluvia)
        return P_Lluvia[valor]

    elif nombre_var == "SensorAgua":
        lluvia_val = asignacion["Lluvia"]
        return P_SensorAgua_dado_Lluvia[(valor, lluvia_val)]

    elif nombre_var == "Alarma":
        sensor_val = asignacion["SensorAgua"]
        return P_Alarma_dado_Sensor[(valor, sensor_val)]

    else:
        raise ValueError("Variable desconocida: " + nombre_var)

# ---------------------------------------------------------
# 3. Probabilidad conjunta P(asignación_completa)
# ---------------------------------------------------------
def prob_conjunta(asignacion: Dict[str,bool]) -> float:
    """
    Calcula P(L, S, A) = Π_i P(X_i | padres(X_i))
    para una asignación completa, ej:
        {'Lluvia': True, 'SensorAgua': False, 'Alarma': True}
    """
    total = 1.0
    for var in variables:
        total *= prob_variable(var, asignacion[var], asignacion)
    return total

# ---------------------------------------------------------
# 4. Enumeración general:
#    P(query=val | evidencia_dict)
#
#    1) Construye dos numeradores:
#          num_true  = P(query=True , evidencia)
#          num_false = P(query=False, evidencia)
#
#    2) Normaliza:
#          P(True|evid)  = num_true/(num_true+num_false)
#          P(False|evid) = num_false/(num_true+num_false)
#
#    Para cada num_*:
#       - recorremos TODAS las combinaciones de las variables
#         no fijadas todavía (ocultas) y sumamos.
# ---------------------------------------------------------
def enumerar_prob_condicional(query_var: str,
                              evidencia: Dict[str,bool]) -> Dict[bool,float]:
    """
    Regresa un diccionario con:
      {
        True:  P(query_var=True  | evidencia),
        False: P(query_var=False | evidencia)
      }
    usando enumeración de todas las asignaciones posibles.
    """

    # 1. Variables libres/ocultas
    #    = todas las variables menos:
    #      - la query
    #      - las evidencias ya fijadas
    otras_vars = [v for v in variables if v != query_var and v not in evidencia]

    resultados = {}

    for valor_query in [True, False]:
        # Construimos un numerador:
        #   num = Σ_{asignaciones de otras_vars} P(asignación completa consistente)
        num = 0.0

        # Para cada combinación posible de las variables ocultas
        for combinacion in itertools.product([True,False], repeat=len(otras_vars)):
            # Creamos una asignación completa tentativa
            asignacion_completa = {}

            # Primero metemos la evidencia conocida
            for var_evi, val_evi in evidencia.items():
                asignacion_completa[var_evi] = val_evi

            # Metemos la query con el valor actual que estamos probando
            asignacion_completa[query_var] = valor_query

            # Metemos las variables ocultas con la combinación actual
            for var_oculta, val_oculta in zip(otras_vars, combinacion):
                asignacion_completa[var_oculta] = val_oculta

            # Ahora que tenemos una asignación COMPLETA para todas las vars,
            # podemos calcular P(asignación)
            p = prob_conjunta(asignacion_completa)

            num += p

        resultados[valor_query] = num

    # 2. Normalizamos
    normalizador = resultados[True] + resultados[False]
    resultados[True]  /= normalizador
    resultados[False] /= normalizador

    return resultados

# ---------------------------------------------------------
# 5. Ejemplo de consulta:
#    "Si escucho la alarma, ¿qué tan probable es que esté lloviendo?"
#
#    Es decir:
#         P(Lluvia | Alarma=True)
#
#    Aquí:
#        query_var  = "Lluvia"
#        evidencia  = {"Alarma": True}
# ---------------------------------------------------------

print("==============================================")
print("TRACE INFERENCIA POR ENUMERACIÓN")
print("==============================================\n")

query = "Lluvia"
evid = {"Alarma": True}

print("1) Pregunta de inferencia:")
print("   ¿Cuál es P(Lluvia | Alarma=True)?")
print("   Interpretación: 'Suena la alarma. ¿Prob que esté lloviendo?'\n")

posterior = enumerar_prob_condicional(query, evid)

# ---------------------------------------------------------
# 6. Imprimir resultados y desglose
# ---------------------------------------------------------
print("2) Resultado de la inferencia por enumeración:")
print(f"   P(Lluvia=True  | Alarma=True) = {posterior[True]:.4f}")
print(f"   P(Lluvia=False | Alarma=True) = {posterior[False]:.4f}\n")

print("3) Interpretación:")
print("- El algoritmo probó TODAS las combinaciones posibles de las demás")
print("  variables (en este caso 'SensorAgua'), y sumó sus probabilidades.")
print("")
print("- Esto es literalmente 'sumar sobre variables ocultas'.")
print("  Por eso se llama inferencia por enumeración.")
print("")
print("- Este método es exacto pero puede ser caro cuando hay muchas")
print("  variables, porque el número de combinaciones explota (2^n, 3^n, etc).")
print("")
print("- En el siguiente script (11 - Eliminación de Variables)")
print("  vamos a hacer lo MISMO pero de forma más eficiente,")
print("  eliminando variables paso a paso sin repetir trabajo.\n")

# Nota:
# - Aquí tampoco generamos gráfica, porque lo central es la suma
#   exhaustiva sobre asignaciones y la normalización al final.
# - Más adelante, cuando lleguemos a los métodos de muestreo
#   (12, 13, 14), sí puede tener sentido graficar histogramas
#   de frecuencias empíricas vs probabilidades teóricas.
