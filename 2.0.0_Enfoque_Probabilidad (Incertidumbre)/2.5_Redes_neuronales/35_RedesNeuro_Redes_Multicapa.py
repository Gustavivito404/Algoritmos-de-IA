# =========================================================
# 35 - REDES MULTICAPA (MLP - Multilayer Perceptron)
# ---------------------------------------------------------
# Descripción:
#   Este script muestra cómo una red neuronal con
#   **una capa oculta** (MLP) puede resolver el problema
#   **XOR**, que el perceptrón clásico no puede aprender.
#
#   Conceptos clave:
#     • Capas ocultas → combinan características intermedias.
#     • Activaciones no lineales → permiten separar regiones curvas.
#     • Aprendizaje supervisado → ajuste de pesos con backpropagation.
#
#   Arquitectura usada:
#       Entrada (2 neuronas)
#           ↓
#       Capa Oculta (4 neuronas, activación tanh)
#           ↓
#       Capa de Salida (1 neurona, activación sigmoide)
#
#   Librerías utilizadas:
#       - NumPy  (cálculo matricial)
#       - Matplotlib (visualización de la frontera)
#
#   Resultado esperado:
#       El modelo aprenderá correctamente el patrón XOR,
#       mostrando una frontera de decisión curva que separa
#       las esquinas opuestas.
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Dataset XOR (no linealmente separable)
# ---------------------------------------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)

# ---------------------------------------------------------
# 2. Parámetros de la red
# ---------------------------------------------------------
input_dim = 2
hidden_dim = 4
output_dim = 1
lr = 0.1
EPOCHS = 5000

# Inicialización aleatoria de pesos
np.random.seed(42)
W1 = np.random.randn(input_dim, hidden_dim)
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim)
b2 = np.zeros((1, output_dim))

# ---------------------------------------------------------
# 3. Funciones de activación y derivadas
# ---------------------------------------------------------
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoide(x):
    return sigmoide(x) * (1 - sigmoide(x))

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x)**2

# ---------------------------------------------------------
# 4. Entrenamiento (Forward + Backprop)
# ---------------------------------------------------------
losses = []
for epoch in range(EPOCHS):
    # Forward pass
    z1 = X @ W1 + b1
    a1 = tanh(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoide(z2)

    # Cálculo del error
    error = y - a2
    loss = np.mean(error**2)
    losses.append(loss)

    # Backpropagation
    d_a2 = -2 * error
    d_z2 = d_a2 * d_sigmoide(z2)
    d_W2 = a1.T @ d_z2
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)

    d_a1 = d_z2 @ W2.T
    d_z1 = d_a1 * d_tanh(z1)
    d_W1 = X.T @ d_z1
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)

    # Actualización de pesos
    W2 -= lr * d_W2
    b2 -= lr * d_b2
    W1 -= lr * d_W1
    b1 -= lr * d_b1

# ---------------------------------------------------------
# 5. Resultados finales
# ---------------------------------------------------------
print("==============================================")
print("RESULTADOS RED MULTICAPA (MLP)")
print("==============================================")
print(f"Error final (MSE): {losses[-1]:.6f}")

# Predicciones finales
a2_final = (a2 > 0.5).astype(int)
for i in range(len(X)):
    print(f"Entrada {X[i]} → Predicción: {a2_final[i][0]} (Real: {y[i][0]})")

# ---------------------------------------------------------
# 6. Visualización de la frontera de decisión
# ---------------------------------------------------------
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

# Forward sobre la grilla
z1 = grid @ W1 + b1
a1 = tanh(z1)
z2 = a1 @ W2 + b2
preds = sigmoide(z2).reshape(xx.shape)

plt.figure(figsize=(6,5))
plt.contourf(xx, yy, preds > 0.5, alpha=0.3, cmap="coolwarm")
plt.scatter(X[y[:,0]==0][:,0], X[y[:,0]==0][:,1], color="royalblue", label="Clase 0", edgecolors="k")
plt.scatter(X[y[:,0]==1][:,0], X[y[:,0]==1][:,1], color="tomato", label="Clase 1", edgecolors="k")
plt.title("MLP resolviendo el problema XOR")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

# ---------------------------------------------------------
# 7. Curva de pérdida
# ---------------------------------------------------------
plt.figure(figsize=(6,3))
plt.plot(losses, color="purple")
plt.title("Evolución del Error (MSE)")
plt.xlabel("Épocas")
plt.ylabel("Error cuadrático medio")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
