# =========================================================
# 03 - PROBABILIDAD CONDICIONADA Y NORMALIZACIÓN
# ---------------------------------------------------------
# Descripción:
#   En este script vamos a:
#     1) Calcular una probabilidad CONDICIONADA.
#     2) Aplicar NORMALIZACIÓN para obtener una distribución válida.
#     3) Visualizar cómo cambian las creencias "antes vs después"
#        de observar evidencia (incluye código de gráfica).
#
#   Ejemplo usado:
#     Tenemos un sensor que detecta "lluvia".
#     El sensor dice: "Sí, está lloviendo".
#     Pregunta: ¿Cuál es la probabilidad real de que SÍ esté lloviendo?
#
#   Eso es P(lluvia | sensor='lluvia')
#
#   Ojo: esto NO es lo mismo que P(sensor='lluvia' | lluvia).
#   Ahí está la magia de la probabilidad condicionada.
#
# ---------------------------------------------------------
# Características:
#   - La probabilidad condicionada mide:
#        "¿Cuál es la prob de A si ya sé que pasó B?"
#
#   - La normalización asegura que las probabilidades
#     finales sumen 1 y formen una distribución válida.
#
# ---------------------------------------------------------
# Fórmulas clave:
#
# (1) Probabilidad condicionada:
#     P(A | B) = P(A ∧ B) / P(B)
#
#     donde:
#       P(A ∧ B) = P(B | A) * P(A)
#
# (2) Regla de Bayes (versión informal):
#     P(lluvia | sensor='lluvia')
#        ∝ P(sensor='lluvia' | lluvia) * P(lluvia)
#
#     Luego NORMALIZAMOS para que:
#        P(lluvia | sensor='lluvia') +
#        P(~lluvia | sensor='lluvia') = 1
#
# ---------------------------------------------------------
# Ejemplo narrativo:
#
#   El robot meteorológico tiene:
#
#   - Creencia inicial (a priori):
#       P(lluvia) = 0.3
#       P(~lluvia) = 0.7
#
#   - Confiabilidad del sensor:
#       P(sensor='lluvia' | lluvia) = 0.9      (detecta lluvia bien)
#       P(sensor='lluvia' | ~lluvia) = 0.2     (a veces se equivoca)
#
#   Observación:
#       El sensor dice "lluvia".
#
#   Pregunta:
#       ¿Le creo? ¿Qué tan seguro estoy ahora de que llueve?
#
# =========================================================

from typing import Dict
import matplotlib.pyplot as plt  # usamos matplotlib para graficar barras comparativas

# ---------------------------------------------------------
# 1. Datos del problema (mismo contexto que 01 y 02)
# ---------------------------------------------------------

# Probabilidad a priori de que llueva:
P_lluvia = 0.3                 # P(lluvia)
P_no_lluvia = 1 - P_lluvia     # P(~lluvia) = 0.7

# Confiabilidad del sensor:
P_sensor_lluvia_given_lluvia = 0.9   # P(sensor='lluvia' | lluvia)
P_sensor_lluvia_given_no = 0.2       # P(sensor='lluvia' | ~lluvia)

# Observación actual:
# El sensor reportó "lluvia"
observacion = "sensor='lluvia'"

print("==============================================")
print("TRACE PROBABILIDAD CONDICIONADA Y NORMALIZACIÓN")
print("==============================================\n")

print("1) Conocimiento previo del agente (A PRIORI):")
print(f"   P(lluvia)     = {P_lluvia:.2f}")
print(f"   P(~lluvia)    = {P_no_lluvia:.2f}\n")

print("2) Modelo del sensor:")
print(f"   P(sensor='lluvia' | lluvia)     = {P_sensor_lluvia_given_lluvia:.2f}")
print(f"   P(sensor='lluvia' | ~lluvia)    = {P_sensor_lluvia_given_no:.2f}\n")

print("3) Evidencia observada:")
print(f"   Observamos -> {observacion}")
print("   Queremos -> P(lluvia | sensor='lluvia')\n")

# ---------------------------------------------------------
# 2. Paso 1: prob conjunta aproximada (sin normalizar)
# ---------------------------------------------------------
# La idea:
#   Cuánta "confianza bruta" tengo en cada hipótesis,
#   ponderada por el sensor.
#
#   score_llueve =
#       P(sensor='lluvia' | lluvia) * P(lluvia)
#
#   score_no_llueve =
#       P(sensor='lluvia' | ~lluvia) * P(~lluvia)
#
# Nota:
#   Estos "scores" NO suman 1 todavía.
#   Todavía no son probabilidades verdaderas.
#   Son pesos relativos que dicen:
#      "qué tan creíble es cada explicación de la evidencia".
# ---------------------------------------------------------

score_llueve = P_sensor_lluvia_given_lluvia * P_lluvia
score_no_llueve = P_sensor_lluvia_given_no * P_no_lluvia

print("4) Pesos sin normalizar (verosimilitudes ponderadas):")
print(f"   score(lluvia)     = P(sensor='lluvia'|lluvia)*P(lluvia)")
print(f"                     = {P_sensor_lluvia_given_lluvia:.2f} * {P_lluvia:.2f}")
print(f"                     = {score_llueve:.4f}")
print()
print(f"   score(~lluvia)    = P(sensor='lluvia'|~lluvia)*P(~lluvia)")
print(f"                     = {P_sensor_lluvia_given_no:.2f} * {P_no_lluvia:.2f}")
print(f"                     = {score_no_llueve:.4f}\n")

# ---------------------------------------------------------
# 3. Paso 2: normalización
# ---------------------------------------------------------
# Ahora convertimos esos "scores" en probabilidades reales.
#
#   P(lluvia | sensor='lluvia') =
#       score_llueve / (score_llueve + score_no_llueve)
#
#   P(~lluvia | sensor='lluvia') =
#       score_no_llueve / (score_llueve + score_no_llueve)
#
# Este denominador:
#       Z = score_llueve + score_no_llueve
# se conoce como constante de normalización.
# ---------------------------------------------------------

Z = score_llueve + score_no_llueve  # constante de normalización

posterior_lluvia = score_llueve / Z
posterior_no_lluvia = score_no_llueve / Z

print("5) Normalización:")
print(f"   Z = score(lluvia) + score(~lluvia)")
print(f"     = {score_llueve:.4f} + {score_no_llueve:.4f}")
print(f"     = {Z:.4f}\n")

print("6) Probabilidades POSTERIORES (ya normalizadas):")
print(f"   P(lluvia | sensor='lluvia')   = {posterior_lluvia:.3f}")
print(f"   P(~lluvia | sensor='lluvia')  = {posterior_no_lluvia:.3f}\n")

print("   Nota importante:")
print("   Antes pensábamos que llover era 30%.")
print("   Después de oír al sensor decir 'lluvia',")
print("   ahora creemos que llover es ~66-67%.\n")

# ---------------------------------------------------------
# 4. Resumen intuitivo
# ---------------------------------------------------------
print("RESUMEN INTUITIVO:")
print("- La probabilidad condicionada responde:")
print("    'Dado que pasó X, ¿qué tan probable es Y?'")
print()
print("- La normalización garantiza que las nuevas creencias")
print("  formen una distribución válida (suman 1).")
print()
print("- Este proceso ES básicamente la Regla de Bayes en acción,")
print("  que formalizaremos en el script 06.\n")

# ---------------------------------------------------------
# 5. Visualización - Comparación A Priori vs Posterior
# ---------------------------------------------------------
# Aquí es donde una gráfica ayuda MUCHO:
#
#   Barras lado a lado:
#   - Antes de la evidencia (a priori)
#   - Después de la evidencia (posterior)
#
#   Así puedes ver visualmente cuánto aumentó la confianza
#   en que sí está lloviendo cuando el sensor dijo "lluvia".
#
#   Vamos a dejar el código listo para ejecutar con matplotlib.
#   La gráfica NO es necesaria para la lógica matemática,
#   pero es excelente para reportes / presentaciones.
# ---------------------------------------------------------

labels = ['lluvia', 'no lluvia']

# Distribución A PRIORI (antes de observar el sensor)
a_priori = [P_lluvia, P_no_lluvia]

# Distribución POSTERIOR (después de observar sensor='lluvia')
posterior = [posterior_lluvia, posterior_no_lluvia]

x = [0, 1]  # posiciones base en el eje X

# offset para separar las barras de cada grupo
offset = 0.2

plt.figure(figsize=(6,4))
# barras de a priori
plt.bar([xi - offset for xi in x],
        a_priori,
        width=0.4,
        label='A priori')

# barras de posterior
plt.bar([xi + offset for xi in x],
        posterior,
        width=0.4,
        label='Posterior\n(dado sensor="lluvia")')

plt.xticks(x, labels)
plt.ylim(0,1)
plt.ylabel("Probabilidad")
plt.title("Actualización de creencias\npor evidencia del sensor")
plt.legend()
plt.show()

# MUY IMPORTANTE para uso interactivo:
# plt.show()
#
# Nota sobre plt.show():
# - En un entorno normal de Python (local, Jupyter, etc.)
#   plt.show() abre la ventana/gráfico.
# - Aquí lo dejo comentado para que el script siga siendo
#   autocontenido sin forzar entorno gráfico.
# - Cuando quieras verlo tú en tu máquina, simplemente
#   descomenta plt.show().

# Nota:
# - Esta gráfica te deja VER el efecto de la evidencia:
#     la barra "lluvia" sube,
#     la barra "no lluvia" baja.
#
# - Esto ayuda a explicar visualmente la diferencia entre:
#     creencia inicial vs creencia actualizada.
#
# - En 04 (Distribución de Probabilidad) también puede ser
#   útil graficar porque ahí hablamos de "toda la masa de prob".
