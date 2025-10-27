# =========================================================
# 08 - REGLA DE LA CADENA
# ---------------------------------------------------------
# Descripción:
#   Este script muestra la "Regla de la Cadena" (Chain Rule)
#   aplicada a una Red Bayesiana pequeña.
#
#   La regla de la cadena nos deja escribir la probabilidad
#   conjunta de TODAS las variables como un producto de
#   probabilidades condicionales más pequeñas.
#
#   En una Red Bayesiana esto se vuelve MUY cómodo porque
#   cada variable solo depende de sus padres en el grafo.
#
# Características:
#   - Convierte P(X1, X2, X3, ...) en factores locales.
#   - Reduce la necesidad de tablas gigantes.
#   - Es la base de toda inferencia en redes Bayesianas.
#
# Ejemplo visual (misma red que 07):
#
#     Lluvia  ──►  SensorAgua  ──►  Alarma
#
#   Eso implica:
#
#     P(Lluvia, SensorAgua, Alarma)
#       = P(Lluvia)
#       * P(SensorAgua | Lluvia)
#       * P(Alarma | SensorAgua)
#
# Fórmula general (para variables X1..Xn):
#
#     P(X1, X2, ..., Xn)
#       = Π_i  P(Xi | padres(Xi))
#
#   donde "padres(Xi)" son los nodos que apuntan a Xi en la red.
#
# En este script vamos a:
#   1) Reusar la red bayesiana de "Lluvia -> SensorAgua -> Alarma".
#   2) Construir paso a paso la prob conjunta usando la regla.
#   3) Verificar que la suma sobre todas las combinaciones = 1.
# =========================================================

from typing import Dict, Tuple
import itertools

# ---------------------------------------------------------
# 1. Definimos la misma red bayesiana del script 07
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
# 2. Función de probabilidad conjunta según la regla de la cadena
# ---------------------------------------------------------
# Aquí aplicamos literalmente:
#
#   P(L, S, A) = P(L) * P(S|L) * P(A|S)
#
# donde:
#   L = Lluvia
#   S = SensorAgua
#   A = Alarma
#
# Nota:
#   Esta es la Regla de la Cadena "adaptada" a la estructura
#   de una Red Bayesiana. En una red general sería:
#       producto sobre nodos de P(nodo | padres_del_nodo)
#
def prob_conjunta(lluvia: bool, sensor: bool, alarma: bool) -> float:
    pL = P_Lluvia[lluvia]
    pS = P_SensorAgua_dado_Lluvia[(sensor, lluvia)]
    pA = P_Alarma_dado_Sensor[(alarma, sensor)]
    return pL * pS * pA

# ---------------------------------------------------------
# 3. Enumerar todas las combinaciones posibles y mostrarlas
# ---------------------------------------------------------
print("==============================================")
print("TRACE REGLA DE LA CADENA")
print("==============================================\n")

print("1) Estructura causal usada:")
print("   Lluvia  ──►  SensorAgua  ──►  Alarma\n")

print("2) Regla de la Cadena en esta red:")
print("   P(L, S, A) = P(L) * P(S|L) * P(A|S)")
print("   donde:")
print("     L = Lluvia")
print("     S = SensorAgua")
print("     A = Alarma\n")

print("3) Probabilidades conjuntas por cada combinación (L,S,A):\n")

todas_comb = list(itertools.product([True, False], repeat=3))
suma_total = 0.0

for (L, S, A) in todas_comb:
    p = prob_conjunta(L, S, A)
    suma_total += p

    # Impresión detallada paso a paso
    print(f"   Caso: Lluvia={L}, SensorAgua={S}, Alarma={A}")
    print(f"     P(Lluvia={L})                 = {P_Lluvia[L]:.3f}")
    print(f"     P(SensorAgua={S} | Lluvia={L}) = {P_SensorAgua_dado_Lluvia[(S,L)]:.3f}")
    print(f"     P(Alarma={A} | SensorAgua={S}) = {P_Alarma_dado_Sensor[(A,S)]:.3f}")
    print(f"     Producto total (conjunta)      = {p:.6f}\n")

print(f"Suma de TODAS las conjuntas = {suma_total:.6f}")
print("   (Esto debe ser ≈ 1.0, confirma que la red define")
print("    una distribución de probabilidad válida sobre el mundo.)\n")

# ---------------------------------------------------------
# 4. Interpretación intuitiva
# ---------------------------------------------------------
print("4) Interpretación:")
print("- La 'Regla de la Cadena' nos deja factorizar la distribución")
print("  conjunta grande en pedacitos locales que sí sabemos modelar.")
print("")
print("- En lugar de tener que almacenar P(L,S,A) como una tabla de 2x2x2")
print("  (8 números explícitos), la red dice:")
print("      'solo necesito:'")
print("        • P(L)")
print("        • P(S|L)")
print("        • P(A|S)")
print("")
print("- Eso escala MUCHO MEJOR cuando tienes 10, 20, 100 variables.")
print("  Porque cada variable solo depende de sus padres, no de todas las demás.\n")

# ---------------------------------------------------------
# 5. Conexión con lo que sigue
# ---------------------------------------------------------
print("5) ¿Por qué esto importa para lo siguiente?")
print("- El siguiente algoritmo (09 - Manto de Markov) va a responder:")
print("    '¿De qué variables depende DIRECTAMENTE una variable?'")
print("  Eso nos dice qué información necesitamos para razonar sobre una variable")
print("  sin tener que mirar TODO el resto de la red.")
print("")
print("- Ese concepto (Manto de Markov) es FUNDAMENTAL para inferencia eficiente,")
print("  para muestreo tipo Gibbs, y para reducir cálculos en redes grandes.\n")

# Nota:
#   - Aquí NO usamos gráfica nueva, porque la regla de la cadena
#     se entiende mejor con el producto paso a paso.
#   - Podemos reusar el dibujito matplotlib del script 07 si quieres
#     visualizar otra vez la estructura causal.
