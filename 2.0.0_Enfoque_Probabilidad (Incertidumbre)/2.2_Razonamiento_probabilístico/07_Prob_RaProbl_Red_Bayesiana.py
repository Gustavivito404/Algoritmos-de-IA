# =========================================================
# 07 - RED BAYESIANA
# ---------------------------------------------------------
# Descripción:
#   En este script construimos una Red Bayesiana MUY sencilla
#   y la usamos para razonar sobre probabilidades.
#
#   Una Red Bayesiana es:
#     - Un grafo dirigido acíclico (DAG).
#     - Cada nodo = variable aleatoria.
#     - Cada flecha = "causa probable" o "influencia directa".
#     - Cada nodo tiene una tabla P(nodo | padres).
#
#   Esto nos permite construir una distribución conjunta
#   de TODO el mundo como un producto de términos locales.
#
# Características:
#   - Captura relaciones causales tipo:
#         Clima --> Sensor --> Lectura
#   - Nos deja hacer inferencia: actualizar creencias
#     cuando observamos evidencia parcial.
#
# Ejemplo visual (conceptual):
#
#     Lluvia  ──►  SensorDetecta  ──►  Alarma
#
#   Interpretación:
#     - Si llueve, es más probable que el sensor detecte "agua".
#     - Si el sensor detecta algo, es más probable que dispare alarma.
#
# Fórmulas clave:
#   Probabilidad conjunta de TODAS las variables:
#
#     P(Lluvia, Sensor, Alarma)
#       = P(Lluvia)
#       * P(Sensor | Lluvia)
#       * P(Alarma | Sensor)
#
#   Nota:
#     Cada factor depende sólo de sus "padres" en el grafo.
#     Eso es lo que hace escalable a las Redes Bayesianas.
#
# =========================================================

from typing import Dict, Tuple
import itertools
import matplotlib.pyplot as plt  # <-- usaremos matplotlib SOLO en la parte comentada final

# ---------------------------------------------------------
# 1. Definición de las variables de la red
# ---------------------------------------------------------
# Vamos a usar 3 variables booleanas:
#
#   Lluvia        = ¿está lloviendo?            (True/False)
#   SensorAgua    = ¿el sensor detecta humedad? (True/False)
#   Alarma        = ¿se dispara la alarma?      (True/False)
#
# Estructura causal (DAG):
#
#   Lluvia ──► SensorAgua ──► Alarma
#
# Padres(Lluvia)      = ninguno
# Padres(SensorAgua)  = {Lluvia}
# Padres(Alarma)      = {SensorAgua}
#
# Esta red dice:
#   - La lluvia influye al sensor.
#   - El sensor influye a la alarma.
#
# Nota:
#   Esto es una simplificación tipo domótica / clima.
#

# Probabilidades base (estas son las tablas de la red bayesiana)

# Nodo raíz: P(Lluvia)
P_Lluvia: Dict[bool, float] = {
    True: 0.3,   # P(Lluvia=True)
    False: 0.7   # P(Lluvia=False)
}

# Nodo SensorAgua depende de Lluvia:
# P(SensorAgua | Lluvia)
P_SensorAgua_dado_Lluvia: Dict[Tuple[bool, bool], float] = {
    # clave: (SensorAgua, Lluvia)
    (True,  True): 0.9,   # Si llueve, el sensor marca agua con 90%
    (False, True): 0.1,

    (True,  False): 0.2,  # Si NO llueve, el sensor a veces se confunde (20% falso positivo)
    (False, False): 0.8
}

# Nodo Alarma depende de SensorAgua:
# P(Alarma | SensorAgua)
P_Alarma_dado_Sensor: Dict[Tuple[bool, bool], float] = {
    # clave: (Alarma, SensorAgua)
    (True,  True): 0.95,  # Si el sensor dice "hay agua", alarma casi siempre suena
    (False, True): 0.05,

    (True,  False): 0.01, # Si el sensor NO detecta agua, alarma casi nunca suena
    (False, False): 0.99
}

print("==============================================")
print("TRACE RED BAYESIANA")
print("==============================================\n")

print("1) Estructura causal de la red:")
print("   Lluvia  ──►  SensorAgua  ──►  Alarma\n")

print("2) Tablas de probabilidad locales (CPDs):")
print("   P(Lluvia):")
for v, p in P_Lluvia.items():
    print(f"      P(Lluvia={v}) = {p:.2f}")
print("")

print("   P(SensorAgua | Lluvia):")
print("      Lluvia  SensorAgua  Prob")
for (sensor, lluvia), prob in P_SensorAgua_dado_Lluvia.items():
    print(f"      {lluvia!s:6s} {sensor!s:11s} {prob:.2f}")
print("")

print("   P(Alarma | SensorAgua):")
print("      SensorAgua  Alarma  Prob")
for (alarma, sensor), prob in P_Alarma_dado_Sensor.items():
    print(f"      {sensor!s:11s} {alarma!s:6s} {prob:.2f}")
print("")

# ---------------------------------------------------------
# 2. Probabilidad conjunta completa P(Lluvia, SensorAgua, Alarma)
# ---------------------------------------------------------
# Fórmula general:
#
#   P(L, S, A) =
#       P(L) *
#       P(S | L) *
#       P(A | S)
#
# Vamos a generar TODAS las combinaciones posibles de (L,S,A)
# y calcular su probabilidad conjunta.
#
# Eso nos da la "distribución conjunta" del mundo.
#

def conjunta(lluvia: bool, sensor: bool, alarma: bool) -> float:
    """Calcula P(L=lluvia, S=sensor, A=alarma) usando la red bayesiana."""
    # P(L)
    pL = P_Lluvia[lluvia]

    # P(S | L)
    pS = P_SensorAgua_dado_Lluvia[(sensor, lluvia)]

    # P(A | S)
    pA = P_Alarma_dado_Sensor[(alarma, sensor)]

    return pL * pS * pA

print("3) Probabilidad conjunta P(L, S, A):")
print("   (Mostrando todas las combinaciones posibles)\n")

todas_comb = list(itertools.product([True, False], repeat=3))
suma_total = 0.0
for (L, S, A) in todas_comb:
    p = conjunta(L, S, A)
    suma_total += p
    print(f"   P(Lluvia={L}, SensorAgua={S}, Alarma={A}) = {p:.5f}")

print(f"\n   Suma total de TODAS las conjuntas = {suma_total:.5f}")
print("   Debería ser ≈ 1.0 si todo está bien definido.\n")

# ---------------------------------------------------------
# 3. Ejemplo de inferencia sencilla:
#    Queremos P(Alarma=True)
# ---------------------------------------------------------
# P(Alarma=True) = Σ_{L,S} P(L,S,Alarma=True)
#
# Es decir: sumamos todas las combinaciones consistentes
# donde Alarma=True sin importar L ni S.

def marginal_alarma_true() -> float:
    total = 0.0
    for (L, S, A) in todas_comb:
        if A is True:
            total += conjunta(L,S,A)
    return total

P_Alarma_true = marginal_alarma_true()
print("4) Inferencia marginal:")
print(f"   P(Alarma=True) = {P_Alarma_true:.5f}\n")

# ---------------------------------------------------------
# 4.5 Ejemplo de inferencia condicional tipo Bayes:
#     ¿Cuál es la probabilidad de que esté lloviendo
#     si escuchamos la alarma?
#
#     Queremos: P(Lluvia=True | Alarma=True)
#
#     Por definición:
#       P(L|A) = P(L ∧ A) / P(A)
#
#     Y:
#       P(L ∧ A) = Σ_S P(L,S,A)
#
def posterior_lluvia_dada_alarma() -> float:
    # numerador = P(L=True, A=True)
    num = 0.0
    den = 0.0

    for (L, S, A) in todas_comb:
        p = conjunta(L,S,A)
        if A is True:
            den += p          # esto suma P(A=True)
            if L is True:
                num += p      # esto suma P(L=True AND A=True)

    return num / den

post_lluvia_si_alarma = posterior_lluvia_dada_alarma()

print("5) Inferencia condicional (estilo Bayes con la red):")
print("   Pregunta: Si escucho la Alarma=True, ¿qué tan probable es que esté lloviendo?")
print(f"   Resultado: P(Lluvia=True | Alarma=True) = {post_lluvia_si_alarma:.3f}\n")

# ---------------------------------------------------------
# 5. Interpretación final
# ---------------------------------------------------------
print("INTERPRETACIÓN:")
print("- Una Red Bayesiana define una distribución conjunta grande")
print("  usando factores locales pequeños P(nodo | padres).")
print("")
print("- Podemos calcular:")
print("    • Probabilidades marginales (ej. P(Alarma=True))")
print("    • Probabilidades condicionales tipo Bayes (ej. P(Lluvia|Alarma))")
print("")
print("- Este es el corazón del razonamiento probabilístico en IA.")
print("- Los siguientes temas (Regla de la Cadena, Manto de Markov, etc.)")
print("  van a desmenuzar estas ideas y optimizarlas computacionalmente.\n")


# ---------------------------------------------------------
# 6. (OPCIONAL) Dibujar el grafo de la Red Bayesiana con matplotlib
# ---------------------------------------------------------
# IMPORTANTE:
#   - Esta parte está COMENTADA para que el script corra sin necesidad
#     de entorno gráfico.
#   - Si quieres ver el diagrama, descomenta el bloque plt.* y el plt.show().
#
#   Qué haremos:
#     - Posicionar nodos en el plano
#     - Dibujar flechas Lluvia -> SensorAgua -> Alarma
#     - Etiquetar nodos
#
#   No usamos networkx todavía para mantenerlo simple.

# plt.figure(figsize=(5,2.5))
#
# # Coordenadas manuales de cada nodo en el plano (x,y)
# x_lluvia, y_lluvia = 0.1, 0.5
# x_sensor, y_sensor = 0.5, 0.5
# x_alarma, y_alarma = 0.9, 0.5
#
# # Dibujar nodos como textos con recuadro
# plt.text(x_lluvia, y_lluvia, "Lluvia",
#          ha='center', va='center',
#          bbox=dict(boxstyle="round", fc="lightblue"))
#
# plt.text(x_sensor, y_sensor, "SensorAgua",
#          ha='center', va='center',
#          bbox=dict(boxstyle="round", fc="lightgreen"))
#
# plt.text(x_alarma, y_alarma, "Alarma",
#          ha='center', va='center',
#          bbox=dict(boxstyle="round", fc="salmon"))
#
# # Dibujar flechas (Lluvia -> SensorAgua -> Alarma)
# plt.annotate("",
#              xy=(x_sensor-0.07, y_sensor),
#              xytext=(x_lluvia+0.07, y_lluvia),
#              arrowprops=dict(arrowstyle="->"))
#
# plt.annotate("",
#              xy=(x_alarma-0.07, y_alarma),
#              xytext=(x_sensor+0.07, y_sensor),
#              arrowprops=dict(arrowstyle="->"))
#
# # Limitar ejes para que se vea limpio
# plt.xlim(0,1)
# plt.ylim(0,1)
# plt.axis('off')
# plt.title("Red Bayesiana: Lluvia → SensorAgua → Alarma")
#
# # Mostrar
# # plt.show()

# Nota:
# - Ese dibujito ayuda mucho cuando expliques causalidad:
#   "esta flecha significa que uno afecta probabilísticamente al otro".
# - Podemos reutilizar ese esquema visual para las siguientes redes
#   más grandes (por ejemplo cuando agreguemos 'Humo', 'Incendio', etc.).