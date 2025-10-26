# =========================================================
# 02 - PROBABILIDAD A PRIORI
# ---------------------------------------------------------
# Descripción:
#   En este script exploramos el concepto de
#   "Probabilidad a Priori" o "creencia inicial".
#
#   Es la probabilidad que un agente asigna a un evento
#   ANTES de observar cualquier evidencia o dato nuevo.
#
#   Representa el conocimiento previo del sistema.
#
# Características:
#   - Se basa en información histórica o supuestos iniciales.
#   - Es el punto de partida para aplicar la Regla de Bayes.
#   - Siempre se ajusta después de observar evidencia
#     (para convertirse en probabilidad posterior).
#
# Fórmulas base:
#   P(A) + P(¬A) = 1
#
# Ejemplo visual:
#   🌦️ Clima diario:
#     - El robot meteorológico sabe que, en promedio,
#       llueve 3 de cada 10 días.
#     - Esa es su probabilidad a priori:
#         P(lluvia) = 0.3
# =========================================================

# ---------------------------------------------------------
# 1. Definimos la situación inicial
# ---------------------------------------------------------
print("==============================================")
print("TRACE PROBABILIDAD A PRIORI")
print("==============================================\n")

# El agente tiene una creencia inicial sobre el clima
P_lluvia = 0.3
P_no_lluvia = 1 - P_lluvia

print(f"P(lluvia) (a priori)       = {P_lluvia:.2f}")
print(f"P(~lluvia) (a priori)      = {P_no_lluvia:.2f}\n")

# ---------------------------------------------------------
# 2. Interpretación conceptual
# ---------------------------------------------------------
print("INTERPRETACIÓN:")
print("Antes de observar ningún dato, el agente cree que:")
print("- Hay un 30% de probabilidad de que llueva.")
print("- Hay un 70% de probabilidad de que NO llueva.\n")

# ---------------------------------------------------------
# 3. Comparación con otro ejemplo
# ---------------------------------------------------------
print("COMPARACIÓN CON OTRO CONTEXTO:")
print("Ejemplo: Robot médico que diagnostica una enfermedad rara.")
P_enfermedad = 0.01   # 1% de la población
P_sano = 1 - P_enfermedad

print(f"P(enfermedad) (a priori)   = {P_enfermedad:.2f}")
print(f"P(sano) (a priori)         = {P_sano:.2f}\n")

# ---------------------------------------------------------
# 4. Reflexión final
# ---------------------------------------------------------
print("REFLEXIÓN:")
print("- Estas probabilidades NO dependen de observaciones.")
print("- Son el conocimiento base del agente sobre el mundo.")
print("- En los siguientes temas (condicionada y Bayes),")
print("  aprenderemos a actualizar estas creencias al ver evidencia.")
