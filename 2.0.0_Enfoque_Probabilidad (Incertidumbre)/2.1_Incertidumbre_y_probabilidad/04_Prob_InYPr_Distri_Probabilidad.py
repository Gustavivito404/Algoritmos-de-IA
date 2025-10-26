# =========================================================
# 04 - DISTRIBUCIÓN DE PROBABILIDAD
# ---------------------------------------------------------
# Descripción:
#   En este script trabajamos con la idea de una
#   DISTRIBUCIÓN DE PROBABILIDAD discreta.
#
#   Una distribución de probabilidad:
#     - Asigna una probabilidad a CADA estado posible.
#     - Todas las probabilidades son ≥ 0.
#     - La suma total es EXACTAMENTE 1.
#
#   Ejemplo típico en IA:
#     El agente no sabe exactamente en qué estado está,
#     pero mantiene una CREENCIA sobre cada posible estado.
#
#     Eso lo vimos parecido en los POMDP:
#         b(s) = P(estado = s)
#
#   Aquí lo formalizamos y verificamos matemáticamente.
#
# Características:
#   - Nos permite representar incertidumbre sobre múltiples hipótesis.
#   - Es la base para inferencia bayesiana, filtros, seguimiento de estado.
#   - Podemos medir cosas como "cuál es el estado más probable".
#
# Fórmulas clave:
#   Sea S = {s1, s2, s3, ...}
#   Una distribución P cumple:
#
#        ∑ P(si) = 1
#
#   y    0 ≤ P(si) ≤ 1
#
# Ejemplo narrativo:
#
#   Robot localizador:
#   Cree que puede estar en:
#     - s0 = "cuarto Izquierda"
#     - s1 = "cuarto Centro"
#     - s2 = "cuarto Derecha"
#
#   Su creencia actual:
#     P(s0) = 0.1
#     P(s1) = 0.7
#     P(s2) = 0.2
#
#   Eso ES una distribución de probabilidad.
#
# =========================================================

from typing import Dict

# ---------------------------------------------------------
# 1. Definir una distribución de probabilidad discreta
# ---------------------------------------------------------
# Vamos a modelar la creencia de un robot sobre su ubicación.
# Cada entrada es:
#     estado : probabilidad de estar en ese estado

creencia_robot: Dict[str, float] = {
    "izquierda": 0.10,
    "centro":    0.70,
    "derecha":   0.20
}

print("==============================================")
print("TRACE DISTRIBUCIÓN DE PROBABILIDAD")
print("==============================================\n")

print("1) Creencia inicial sobre la posición del robot:")
for estado, prob in creencia_robot.items():
    print(f"   P(estado = {estado:9s}) = {prob:.2f}")
print("")

# ---------------------------------------------------------
# 2. Verificar que es una distribución válida
# ---------------------------------------------------------
# Reglas:
#   - Todas las probabilidades >= 0
#   - Suman 1 (o muy cerquita, por redondeos numéricos)

valores = list(creencia_robot.values())
suma_prob = sum(valores)

todas_no_negativas = all(p >= 0 for p in valores)
suma_uno = abs(suma_prob - 1.0) < 1e-9  # tolerancia numérica

print("2) Validación de la distribución:")
print(f"   ¿Todas las probabilidades son >= 0?   {todas_no_negativas}")
print(f"   Suma total de probabilidades = {suma_prob:.4f}")
print(f"   ¿Suma (≈) 1?                           {suma_uno}\n")

# ---------------------------------------------------------
# 3. Obtener el estado más probable
# ---------------------------------------------------------
# Esto es útil en toma de decisiones:
#   "Según mi creencia actual, ¿dónde ES MÁS probable que esté?"

estado_mas_probable = max(creencia_robot, key=lambda e: creencia_robot[e])
prob_mas_alta = creencia_robot[estado_mas_probable]

print("3) Estado más probable actualmente:")
print(f"   El robot cree más probable estar en: '{estado_mas_probable}'")
print(f"   Con probabilidad: {prob_mas_alta:.2f}\n")

# ---------------------------------------------------------
# 4. Normalización manual (si la suma != 1)
# ---------------------------------------------------------
# En la práctica, a veces la suma NO da 1 exactamente,
# por ruido, errores numéricos, sensores raros, etc.
#
# Ejemplo: imagina una creencia sin normalizar
creencia_sucia: Dict[str, float] = {
    "izquierda": 2.0,
    "centro":    5.0,
    "derecha":   3.0
}
# Esto NO es una distribución todavía,
# son solo "pesos" relativos.

print("4) Ejemplo de 'creencia sucia' (no normalizada):")
for estado, peso in creencia_sucia.items():
    print(f"   peso[{estado:9s}] = {peso:.3f}")
print("   (Estos no son todavía probabilidades válidas)\n")

# Normalizamos:
total_pesos = sum(creencia_sucia.values())
creencia_normalizada = {
    estado: peso / total_pesos
    for estado, peso in creencia_sucia.items()
}

print("   Normalizando esos pesos para convertirlos a P(estado):")
for estado, prob in creencia_normalizada.items():
    print(f"   P_norm({estado:9s}) = {prob:.3f}")
print(f"   Suma total normalizada = {sum(creencia_normalizada.values()):.3f}\n")

# ---------------------------------------------------------
# 5. Interpretación final
# ---------------------------------------------------------
print("INTERPRETACIÓN:")
print("- Una DISTRIBUCIÓN DE PROBABILIDAD es la forma estándar")
print("  en IA para representar incertidumbre sobre múltiples")
print("  estados posibles.")
print("")
print("- La NORMALIZACIÓN garantiza que esos valores se puedan")
print("  interpretar como verdaderas probabilidades.")
print("")
print("- Este mismo truco de normalizar lo usamos en:")
print("    • Filtros Bayesianos")
print("    • POMDPs (creencia del agente)")
print("    • Inferencia en redes Bayesianas")
print("")
print("- En el siguiente script (05 - Independencia Condicional)")
print("  vamos a ver cómo las variables pueden depender (o no)")
print("  entre sí, y cómo eso simplifica MUCHÍSIMO los cálculos.")
print("")
# Nota:
# - Aquí podríamos graficar esta distribución como barras
#   (izquierda / centro / derecha).
# - Esa gráfica ayuda a visualizar qué estado es el más probable.
# - ¿Quieres que también te dé el snippet de matplotlib para esto
#   como en el script anterior? Si quieres lo preparo igual.
