# =========================================================
# 05 - INDEPENDENCIA CONDICIONAL
# ---------------------------------------------------------
# Descripción:
#   En este script exploramos el concepto de
#   INDEPENDENCIA y CONDICIONALIDAD en probabilidad.
#
#   En IA, muchas variables del mundo están relacionadas,
#   pero no todas DEPENDEN directamente unas de otras.
#   Saber cuándo una variable es INDEPENDIENTE de otra
#   nos permite SIMPLIFICAR enormes cálculos.
#
# Características:
#   - Si A y B son independientes:
#         P(A ∧ B) = P(A) * P(B)
#
#   - Si A y B son condicionalmente independientes dado C:
#         P(A ∧ B | C) = P(A | C) * P(B | C)
#
#   Esto es la base para las Redes Bayesianas, donde
#   cada nodo solo depende de sus "padres".
#
# Ejemplo narrativo:
#   🏠 Sistema de alarma inteligente:
#     Variables:
#       R = Lluvia
#       S = Sensor de humo
#       A = Alarma
#
#   - La alarma puede sonar por:
#       • fuego (sensor de humo)
#       • lluvia (falsa alarma por humedad)
#
#   Pero si ya sabemos que el sensor detectó humo,
#   la lluvia se vuelve IRRELEVANTE para explicar la alarma.
#
#   Es decir:
#       P(A | S, R) = P(A | S)
#
# =========================================================

from typing import Dict

# ---------------------------------------------------------
# 1. Definimos probabilidades base
# ---------------------------------------------------------
# Estas son las creencias iniciales sobre cada variable:
#   (usamos ejemplos ficticios para ilustrar dependencias)

P_lluvia = 0.3            # P(R)
P_sensor_humo = 0.05      # P(S)
P_alarma_given_humo = 0.9 # P(A | S)
P_alarma_given_lluvia = 0.2 # P(A | R)
P_alarma_given_no_causa = 0.01 # P(A | ¬S ∧ ¬R)

print("==============================================")
print("TRACE INDEPENDENCIA CONDICIONAL")
print("==============================================\n")

print("1) Probabilidades base:")
print(f"   P(Lluvia)                   = {P_lluvia:.2f}")
print(f"   P(SensorHumo)               = {P_sensor_humo:.2f}")
print(f"   P(Alarma | Humo)            = {P_alarma_given_humo:.2f}")
print(f"   P(Alarma | Lluvia)          = {P_alarma_given_lluvia:.2f}")
print(f"   P(Alarma | ¬Humo ∧ ¬Lluvia) = {P_alarma_given_no_causa:.2f}\n")

# ---------------------------------------------------------
# 2. Caso 1: Independencia simple
# ---------------------------------------------------------
# Supongamos que el sensor de humo y la lluvia son
# completamente INDEPENDIENTES:
#
#   P(S ∧ R) = P(S) * P(R)
#
# Si fueran dependientes, tendríamos que conocer
# una tabla completa P(S ∧ R), que crece exponencialmente.
# ---------------------------------------------------------

P_SyR = P_sensor_humo * P_lluvia

print("2) Caso 1 - Independencia simple:")
print(f"   P(Sensor ∧ Lluvia) = {P_SyR:.4f}")
print("   (porque asumimos independencia entre Sensor y Lluvia)\n")

# ---------------------------------------------------------
# 3. Caso 2: Dependencia condicional
# ---------------------------------------------------------
# Si ahora añadimos una variable "Alarma", las cosas cambian.
#
#   Saber que la alarma sonó cambia las probabilidades
#   de las causas (humo o lluvia).
#
#   Pero si ya sabemos que hubo humo (S),
#   la lluvia (R) deja de aportar información sobre A:
#
#       P(A | S, R) ≈ P(A | S)
#
#   Eso es independencia CONDICIONAL de A respecto a R, dado S.
# ---------------------------------------------------------

print("3) Caso 2 - Independencia condicional:")
P_A_dado_S = P_alarma_given_humo
P_A_dado_SR = P_alarma_given_humo  # Igual, ya que R no cambia A si S=1

print(f"   P(A | S)  = {P_A_dado_S:.2f}")
print(f"   P(A | S,R)= {P_A_dado_SR:.2f}")
print("   -> Son prácticamente iguales, lo que indica que")
print("      A es condicionalmente independiente de R dado S.\n")

# ---------------------------------------------------------
# 4. Comparación conceptual
# ---------------------------------------------------------
print("4) Comparación conceptual:")
print("   Sin condicionar:")
print("     P(A | R)  = 0.2   (lluvia puede causar falsa alarma)")
print("   Condicionado en S:")
print("     P(A | S,R)= 0.9   (si hay humo, la lluvia ya no importa)")
print()
print("   Esto ilustra cómo el conocimiento de una causa")
print("   puede hacer que otras variables se vuelvan irrelevantes.")
print("")

# ---------------------------------------------------------
# 5. Reflexión final
# ---------------------------------------------------------
print("REFLEXIÓN:")
print("- La independencia condicional es la piedra angular")
print("  de las Redes Bayesianas y del razonamiento eficiente.")
print("")
print("- Sin independencia, necesitaríamos tablas de probabilidad")
print("  para todas las combinaciones posibles de variables.")
print("")
print("- Con independencia condicional, solo modelamos las")
print("  dependencias directas, reduciendo exponencialmente")
print("  el tamaño del problema.")
print("")
print("- En el siguiente script (06 - Regla de Bayes), aplicaremos")
print("  estas ideas para ACTUALIZAR creencias al observar evidencia.")
