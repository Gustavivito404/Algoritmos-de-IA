# =========================================================
# 01 - INCERTIDUMBRE
# ---------------------------------------------------------
# Descripción:
#   Este script introduce el concepto fundamental de la
#   INCERTIDUMBRE en Inteligencia Artificial:
#   cómo representar y razonar sobre el mundo cuando
#   no se tiene información completa o segura.
#
#   En IA, la incertidumbre se maneja con PROBABILIDAD.
#   En lugar de afirmar "está lloviendo", decimos:
#       P(llueve) = 0.7
#   Esto significa que el agente asigna una creencia
#   de 70% a la posibilidad de lluvia.
#
# Características:
#   - Representa conocimiento incompleto.
#   - Permite tomar decisiones con riesgo o ruido.
#   - Es base para todos los modelos bayesianos.
#
# Ejemplo visual:
#
#   🌦️ Sensor de lluvia (robot meteorológico)
#   El sensor no es perfecto:
#     - Si llueve, detecta lluvia correctamente el 90% de las veces.
#     - Si NO llueve, da falso positivo el 20% de las veces.
#
#   Meta: representar la incertidumbre del robot sobre si
#         realmente está lloviendo al recibir una lectura "llueve".
#
# Fórmulas base:
#   P(evento) ∈ [0,1]
#   P(no evento) = 1 - P(evento)
#
#   La suma de probabilidades de todos los estados posibles = 1
# =========================================================

# ---------------------------------------------------------
# 1. Definimos el problema
# ---------------------------------------------------------
# El robot tiene dos posibles estados del mundo:
#   - Llueve
#   - No llueve
#
# Y una lectura de su sensor: "detecta lluvia"
# Pero ese sensor no es perfecto.

# Probabilidad a priori de que llueva (basada en el clima histórico)
P_lluvia = 0.3           # 30% de los días llueve
P_no_lluvia = 1 - P_lluvia

# Fiabilidad del sensor
P_sensor_detecta_lluvia_si_llueve = 0.9     # 90% de aciertos
P_sensor_detecta_lluvia_si_no_llueve = 0.2  # 20% de falsos positivos

# ---------------------------------------------------------
# 2. Representamos la incertidumbre
# ---------------------------------------------------------
print("==============================================")
print("TRACE INCERTIDUMBRE - SENSOR DE LLUVIA")
print("==============================================\n")

print(f"Probabilidad de que llueva (P(lluvia))       = {P_lluvia:.2f}")
print(f"Probabilidad de que NO llueva (P(~lluvia))   = {P_no_lluvia:.2f}\n")

print("Confiabilidad del sensor:")
print(f"  P(sensor='lluvia' | lluvia)       = {P_sensor_detecta_lluvia_si_llueve:.2f}")
print(f"  P(sensor='lluvia' | ~lluvia)      = {P_sensor_detecta_lluvia_si_no_llueve:.2f}\n")

# ---------------------------------------------------------
# 3. Calcular la probabilidad total de que el sensor diga "lluvia"
# ---------------------------------------------------------
# Según la regla de la probabilidad total:
#
# P(sensor='lluvia') =
#     P(sensor='lluvia' | lluvia) * P(lluvia)
#   + P(sensor='lluvia' | ~lluvia) * P(~lluvia)
#
# Esto representa la incertidumbre global del sensor.

P_sensor_lluvia = (
    P_sensor_detecta_lluvia_si_llueve * P_lluvia
    + P_sensor_detecta_lluvia_si_no_llueve * P_no_lluvia
)

print(f"Probabilidad total de que el sensor diga 'lluvia': P(sensor='lluvia') = {P_sensor_lluvia:.3f}\n")

# ---------------------------------------------------------
# 4. Interpretación de la incertidumbre
# ---------------------------------------------------------
print("INTERPRETACIÓN:")
print("- El sensor no es determinista.")
print("- Aunque detecte 'lluvia', no estamos 100% seguros de que esté lloviendo.")
print("- Lo que hemos calculado (P(sensor='lluvia')) refleja la incertidumbre general del sistema.\n")

print("Siguientes pasos (en próximos scripts):")
print("  1️⃣ 'Probabilidad a Priori' -> qué creemos ANTES de observar nada.")
print("  2️⃣ 'Probabilidad Condicionada' -> cómo actualizamos esa creencia al ver evidencia.")
print("  3️⃣ 'Regla de Bayes' -> cálculo exacto de P(lluvia | sensor='lluvia').")

# Nota:
# - Este primer ejemplo no hace inferencia, solo cuantifica la
#   incertidumbre en las observaciones del agente.
# - A partir de aquí iremos introduciendo Bayes y normalización.
