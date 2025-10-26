# =========================================================
# 06 - REGLA DE BAYES
# ---------------------------------------------------------
# Descripción:
#   Este script muestra cómo aplicar la Regla de Bayes
#   para actualizar creencias ante nueva evidencia.
#
#   Es el paso más importante de este enfoque:
#   pasar de "probabilidades a priori" a "probabilidades
#   posteriores" después de observar un dato (sensor, test, etc.).
#
# Características:
#   - Usa evidencia observada para ajustar creencias.
#   - Permite inferir causas a partir de efectos.
#   - Es la base de inferencia en Redes Bayesianas,
#     diagnóstico probabilístico y clasificación Naïve Bayes.
#
# Fórmula:
#   P(H | E) = [ P(E | H) * P(H) ] / P(E)
#
#   donde:
#     H = hipótesis   (ej. "está lloviendo")
#     E = evidencia   (ej. "sensor detecta lluvia")
#
#   El denominador P(E) se obtiene con:
#     P(E) = P(E|H) * P(H) + P(E|¬H) * P(¬H)
#
# Ejemplo:
#   Robot meteorológico con sensor imperfecto.
#   Observa que su sensor dice "lluvia".
#   Queremos calcular:
#       P(lluvia | sensor='lluvia')
#
#   Compararemos:
#     - Probabilidad a priori
#     - Probabilidad posterior
#   y mostraremos el cambio con una gráfica.
# =========================================================

import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Datos base del problema
# ---------------------------------------------------------
# Igual que antes, pero ahora aplicamos directamente Bayes.

P_lluvia = 0.3                   # Probabilidad a priori
P_no_lluvia = 1 - P_lluvia

P_sensor_given_lluvia = 0.9      # P(sensor='lluvia' | lluvia)
P_sensor_given_no = 0.2          # P(sensor='lluvia' | ¬lluvia)

# ---------------------------------------------------------
# 2. Aplicar la Regla de Bayes paso a paso
# ---------------------------------------------------------
print("==============================================")
print("TRACE REGLA DE BAYES")
print("==============================================\n")

print("1) Datos iniciales:")
print(f"   P(lluvia)                   = {P_lluvia:.2f}")
print(f"   P(~lluvia)                  = {P_no_lluvia:.2f}")
print(f"   P(sensor='lluvia'|lluvia)   = {P_sensor_given_lluvia:.2f}")
print(f"   P(sensor='lluvia'|~lluvia)  = {P_sensor_given_no:.2f}\n")

# Paso 1: Calcular P(sensor='lluvia') usando la probabilidad total
P_sensor = (
    P_sensor_given_lluvia * P_lluvia
    + P_sensor_given_no * P_no_lluvia
)

print("2) Probabilidad total del sensor:")
print(f"   P(sensor='lluvia') = {P_sensor:.3f}\n")

# Paso 2: Calcular la probabilidad posterior con Bayes
P_lluvia_given_sensor = (
    P_sensor_given_lluvia * P_lluvia
) / P_sensor

P_no_lluvia_given_sensor = (
    P_sensor_given_no * P_no_lluvia
) / P_sensor

print("3) Aplicando la Regla de Bayes:")
print("   P(lluvia | sensor='lluvia') = [ P(sensor|lluvia)*P(lluvia) ] / P(sensor)")
print(f"                                = ({P_sensor_given_lluvia:.2f} * {P_lluvia:.2f}) / {P_sensor:.3f}")
print(f"                                = {P_lluvia_given_sensor:.3f}")
print()
print(f"   P(~lluvia | sensor='lluvia') = {P_no_lluvia_given_sensor:.3f}\n")

# ---------------------------------------------------------
# 3. Comparación A Priori vs Posterior
# ---------------------------------------------------------
print("4) Comparación:")
print(f"   A priori:   P(lluvia)     = {P_lluvia:.2f}")
print(f"   Posterior:  P(lluvia|sensor='lluvia') = {P_lluvia_given_sensor:.2f}")
print()
print("   El sensor aumentó nuestra confianza en que llueve")
print("   de 30% a aproximadamente 66%.")
print("   Eso es EXACTAMENTE lo que hace Bayes: ajustar creencias.\n")

# ---------------------------------------------------------
# 4. Visualización (comparación gráfica)
# ---------------------------------------------------------
# Mostramos cómo cambia la creencia tras observar evidencia.
# Esto es muy ilustrativo en IA: "actualización bayesiana visual".
# ---------------------------------------------------------

labels = ["Lluvia", "No lluvia"]
a_priori = [P_lluvia, P_no_lluvia]
posterior = [P_lluvia_given_sensor, P_no_lluvia_given_sensor]

x = [0, 1]
offset = 0.2

plt.figure(figsize=(6,4))
plt.bar([xi - offset for xi in x], a_priori, width=0.4, label="A priori")
plt.bar([xi + offset for xi in x], posterior, width=0.4, label="Posterior")

plt.xticks(x, labels)
plt.ylim(0,1)
plt.ylabel("Probabilidad")
plt.title("Regla de Bayes: Actualización de creencias\n(sensor detecta lluvia)")
plt.legend()

plt.show()

# ---------------------------------------------------------
# 5. Interpretación final
# ---------------------------------------------------------
print("INTERPRETACIÓN:")
print("- Bayes combina la evidencia con el conocimiento previo.")
print("- Esto permite a un agente aprender de la experiencia.")
print("- En IA se usa en diagnóstico, visión por computadora, NLP, etc.")
print()
print("Ejemplo general:")
print("   P(causa | evidencia) = (P(evidencia | causa) * P(causa)) / P(evidencia)")
print()
print("   • P(causa)  = creencia previa (a priori)")
print("   • P(evidencia|causa) = modelo del mundo o sensor")
print("   • P(causa|evidencia) = creencia actualizada (posterior)\n")

# Nota:
# - Esta gráfica te deja VER cómo las probabilidades cambian.
# - A medida que el sensor se vuelve más confiable, la barra azul
#   (posterior de lluvia) se acerca cada vez más al 1.0.
# - Si el sensor fuera muy malo, ambas barras quedarían parecidas.
# - Este principio será la base del siguiente enfoque:
#     "Razonamiento Probabilístico" (Redes Bayesianas, Inferencia, etc.)
