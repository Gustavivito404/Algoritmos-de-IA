# =========================================================
# 31 - COMPUTACIÓN NEURONAL
# ---------------------------------------------------------
# Descripción:
#   Este script ilustra cómo funciona una neurona artificial
#   individual: toma varias entradas, las pondera con pesos,
#   aplica una suma ponderada, le agrega un sesgo (bias) y
#   pasa el resultado por una función de activación.
#
#   Esta es la "unidad básica" de una red neuronal.
#
# Matemática:
#   y = f( Σ (w_i * x_i) + b )
#
#   Donde:
#     x_i → entradas
#     w_i → pesos sinápticos
#     b   → bias (desplaza la activación)
#     f() → función de activación
#
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Definición de la neurona
# ---------------------------------------------------------
def neurona(x: np.ndarray, w: np.ndarray, b: float, f):
    """
    Calcula la salida de una neurona artificial.

    Parámetros:
        x : vector de entradas
        w : vector de pesos
        b : sesgo (bias)
        f : función de activación (callable)

    Retorna:
        salida de la neurona (float)
    """
    # Paso 1: Suma ponderada
    z = np.dot(w, x) + b
    print(f"Σ(w*x) + b = {z:.4f}")

    # Paso 2: Aplicar función de activación
    y = f(z)
    print(f"f(z) = {y:.4f}")

    return y

# ---------------------------------------------------------
# 2. Ejemplo con entradas y pesos
# ---------------------------------------------------------
# Entradas simuladas (pueden representar señales sensoriales)
x = np.array([0.6, 0.8, 0.3])

# Pesos sinápticos aprendidos
w = np.array([0.5, -0.2, 0.9])

# Bias (ajusta el umbral de activación)
b = 0.1

# ---------------------------------------------------------
# 3. Funciones de activación comunes
# ---------------------------------------------------------
def escalon(z):
    return 1 if z >= 0 else 0

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return max(0, z)

# ---------------------------------------------------------
# 4. Ejecución de ejemplo
# ---------------------------------------------------------
print("==============================================")
print("TRACE COMPUTACIÓN NEURONAL")
print("==============================================")

activaciones = {
    "Escalón": escalon,
    "Sigmoide": sigmoide,
    "Tanh": tanh,
    "ReLU": relu
}

for nombre, f in activaciones.items():
    print(f"\n[{nombre}]")
    salida = neurona(x, w, b, f)

# ---------------------------------------------------------
# 5. Visualización de las funciones de activación
# ---------------------------------------------------------
z = np.linspace(-5, 5, 200)

plt.figure(figsize=(8, 6))
plt.plot(z, [escalon(i) for i in z], label="Escalón", linestyle="--")
plt.plot(z, [sigmoide(i) for i in z], label="Sigmoide")
plt.plot(z, [tanh(i) for i in z], label="Tanh")
plt.plot(z, [relu(i) for i in z], label="ReLU")
plt.title("Funciones de activación neuronales")
plt.xlabel("z (entrada neta)")
plt.ylabel("f(z) (salida)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.show()
