# =========================================================
# 32 - FUNCIONES DE ACTIVACIÓN
# ---------------------------------------------------------
# Descripción:
#   Este script muestra las principales funciones de
#   activación utilizadas en redes neuronales, tanto su
#   forma como su derivada.
#
#   Las funciones de activación definen la salida de una
#   neurona a partir de su entrada neta (suma ponderada),
#   introduciendo **no linealidad** al modelo.
#
#   Aquí se grafican:
#       • Escalón binario
#       • Sigmoide (logística)
#       • Tangente hiperbólica (tanh)
#       • ReLU (Rectified Linear Unit)
#       • Leaky ReLU
#
#   Además, se calcula y muestra su derivada para entender
#   su efecto sobre el gradiente durante el aprendizaje.
#
#   Estas funciones son fundamentales para permitir que las
#   redes multicapa aprendan relaciones no lineales.
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Definición de funciones de activación
# ---------------------------------------------------------

def escalon(x):
    return np.where(x >= 0, 1, 0)

def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

def softplus(x):
    return np.log(1 + np.exp(x))

# ---------------------------------------------------------
# 2. Derivadas (para retropropagación)
# ---------------------------------------------------------
def derivada_sigmoide(x):
    s = sigmoide(x)
    return s * (1 - s)

def derivada_tanh(x):
    return 1 - np.tanh(x)**2

def derivada_relu(x):
    return np.where(x > 0, 1, 0)

def derivada_leaky_relu(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def derivada_softplus(x):
    return sigmoide(x)  # derivada de softplus = sigmoide

# ---------------------------------------------------------
# 3. Rango de valores para evaluar las funciones
# ---------------------------------------------------------
x = np.linspace(-6, 6, 400)

# Diccionario de funciones y derivadas
funciones = {
    "Escalón": (escalon, None),
    "Sigmoide": (sigmoide, derivada_sigmoide),
    "Tanh": (tanh, derivada_tanh),
    "ReLU": (relu, derivada_relu),
    "Leaky ReLU": (leaky_relu, derivada_leaky_relu),
    "Softplus": (softplus, derivada_softplus)
}

# ---------------------------------------------------------
# 4. Visualización de funciones y derivadas
# ---------------------------------------------------------
plt.figure(figsize=(12, 10))
plt.suptitle("Funciones de Activación y sus Derivadas", fontsize=14, fontweight="bold")

for i, (nombre, (f, df)) in enumerate(funciones.items()):
    y = f(x)
    plt.subplot(3, 2, i+1)
    plt.plot(x, y, label=f"{nombre}")
    
    if df is not None:
        dy = df(x)
        plt.plot(x, dy, '--', label="Derivada", color='orange')
    
    plt.title(nombre)
    plt.xlabel("Entrada (z)")
    plt.ylabel("Salida f(z)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ---------------------------------------------------------
# 5. Comentario interpretativo
# ---------------------------------------------------------
print("==============================================")
print("ANÁLISIS DE FUNCIONES DE ACTIVACIÓN")
print("==============================================")
print("""
🔹 Escalón:
   - Salta de 0 a 1. No tiene derivada útil (discontinua).
   - Solo sirve para redes muy simples como el perceptrón clásico.

🔹 Sigmoide:
   - Suaviza la salida entre 0 y 1.
   - Buena para probabilidades, pero su derivada se aplana (problema del gradiente).

🔹 Tanh:
   - Similar a Sigmoide pero centrada en 0 (-1 a 1).
   - Tiende a converger mejor.

🔹 ReLU:
   - Rápida y eficiente (usa solo valores positivos).
   - Es la más usada en redes profundas (CNN, MLP).
   - Derivada simple (0 o 1).

🔹 Leaky ReLU:
   - Variante de ReLU que evita que la neurona “muera”
     permitiendo un pequeño gradiente negativo.

🔹 Softplus:
   - Suaviza la ReLU, derivada = sigmoide.
   - Común en redes probabilísticas o autoencoders.
""")
