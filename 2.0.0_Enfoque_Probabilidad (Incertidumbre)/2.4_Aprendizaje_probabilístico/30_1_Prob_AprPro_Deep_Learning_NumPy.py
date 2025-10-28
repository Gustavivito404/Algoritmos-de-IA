# =========================================================
# 30 - APRENDIZAJE PROFUNDO (RED NEURONAL FEEDFORWARD BÁSICA)
# ---------------------------------------------------------
# Descripción:
#   Entrenamos una red neuronal totalmente conectada (MLP)
#   desde cero usando NumPy, SIN usar frameworks.
#
#   Objetivo:
#       Clasificar puntos 2D en dos clases (0 / 1)
#       aprendiendo una frontera NO LINEAL.
#
#   Arquitectura:
#       Capa oculta: 2 neuronas (activación ReLU)
#       Capa salida: 1 neurona (activación sigmoide)
#
#   Flujo forward:
#       z1 = W1·x + b1
#       a1 = ReLU(z1)
#       z2 = W2·a1 + b2
#       y_hat = sigm(z2)
#
#   Función de pérdida:
#       Binary Cross-Entropy
#
#   Actualización:
#       Gradiente descendente
#
#   Nota:
#       Este script te enseña literalmente cómo "aprende"
#       una red neuronal: propagación hacia adelante,
#       retropropagación y ajuste de pesos.
#
# =========================================================

import numpy as np

# ---------------------------------------------------------
# 1. Funciones de activación y utilidades
# ---------------------------------------------------------
def sigmoid(x):
    """σ(x) = 1 / (1 + e^-x)  -> salida entre 0 y 1"""
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    """Derivada de sigmoid(x). Asumimos que recibimos σ(x) ya calculada."""
    return x * (1 - x)

def relu(x):
    """ReLU(x) = max(0, x) -> no lineal, corta negativos."""
    return np.maximum(0, x)

def drelu(x):
    """Derivada de ReLU: 1 si x>0, 0 si x<=0."""
    grad = np.zeros_like(x)
    grad[x > 0] = 1.0
    return grad

def binary_cross_entropy(y_true, y_pred):
    """
    BCE = - [ y*log(y_hat) + (1-y)*log(1-y_hat) ] promedio.
    y_true y y_pred son vectores columna (N x 1).
    """
    eps = 1e-10  # para evitar log(0)
    return np.mean(-(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps)))

# ---------------------------------------------------------
# 2. Generación de datos de entrenamiento (2 clases en 2D)
# ---------------------------------------------------------
np.random.seed(42)

# Clase 0: nube alrededor de (0,0)
N0 = 50
clase0 = np.random.normal(loc=[0.0, 0.0], scale=0.5, size=(N0, 2))
y0 = np.zeros((N0, 1))

# Clase 1: nube alrededor de (2,2)
N1 = 50
clase1 = np.random.normal(loc=[2.0, 2.0], scale=0.5, size=(N1, 2))
y1 = np.ones((N1, 1))

# Dataset combinado
X = np.vstack([clase0, clase1])       # shape (100, 2)
Y = np.vstack([y0, y1])               # shape (100, 1)

# Barajar (shuffle)
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

# ---------------------------------------------------------
# 3. Inicialización de la red neuronal
# ---------------------------------------------------------
# Tamaños:
#   input_dim = 2   (x,y)
#   hidden_dim = 2  (2 neuronas ocultas)
#   output_dim = 1  (prob clase 1)

input_dim = 2
hidden_dim = 5
output_dim = 1

# Pesos y sesgos iniciales
W1 = np.random.randn(input_dim, hidden_dim) * 0.5  # (2x2)
b1 = np.zeros((1, hidden_dim))                     # (1x2)

W2 = np.random.randn(hidden_dim, output_dim) * 0.5 # (2x1)
b2 = np.zeros((1, output_dim))                     # (1x1)

# Hiperparámetro de entrenamiento
lr = 0.355      # learning rate
EPOCHS = 5000   # iteraciones de entrenamiento

# Para guardar trazas de pérdida
hist_loss = []

# ---------------------------------------------------------
# 4. Entrenamiento (forward + backward + update)
# ---------------------------------------------------------
print("==============================================")
print("TRACE ENTRENAMIENTO RED NEURONAL")
print("==============================================")

for epoch in range(EPOCHS):

    # ---------- FORWARD ----------
    # Capa oculta
    z1 = X @ W1 + b1        # (N x 2)
    a1 = relu(z1)           # (N x 2)

    # Capa salida
    z2 = a1 @ W2 + b2       # (N x 1)
    y_hat = sigmoid(z2)     # (N x 1) probabilidad clase 1

    # Pérdida
    loss = binary_cross_entropy(Y, y_hat)
    hist_loss.append(loss)

    # ---------- BACKWARD ----------
    # Derivada de la pérdida respecto a y_hat:
    # dL/dy_hat = (y_hat - y_true) / (y_hat*(1-y_hat)) en BCE con sigmoid,
    # pero usamos una forma más estable:
    # Para sigmoid + BCE, el gradiente hacia z2 es simplemente:
    # dL/dz2 = (y_hat - Y)
    dL_dz2 = (y_hat - Y)  # (N x 1)

    # Gradientes para W2 y b2
    dL_dW2 = a1.T @ dL_dz2            # (2xN)@(N x1) -> (2x1)
    dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)  # (1x1)

    # Gradiente hacia a1
    dL_da1 = dL_dz2 @ W2.T            # (N x1)@(1x2)->(N x2)

    # Gradiente hacia z1 (pasa por ReLU)
    dL_dz1 = dL_da1 * drelu(z1)       # (N x2) * (N x2)

    # Gradientes para W1 y b1
    dL_dW1 = X.T @ dL_dz1             # (2xN)@(N x2)->(2x2)
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)  # (1x2)

    # ---------- UPDATE (Gradient Descent) ----------
    W2 -= lr * dL_dW2
    b2 -= lr * dL_db2
    W1 -= lr * dL_dW1
    b1 -= lr * dL_db1

    # ---------- TRAZA OCASIONAL ----------
    if epoch % 400 == 0 or epoch == EPOCHS-1:
        # Exactitud aproximada (clasificación dura 0/1)
        pred_clase = (y_hat >= 0.5).astype(int)
        acc = np.mean(pred_clase == Y)

        print(f"[Época {epoch:4d}] loss={loss:.4f}  acc={acc*100:.1f}%")
        print(f"  W1=\n{W1}\n  W2=\n{W2}\n")

# ---------------------------------------------------------
# 5. Evaluación final
# ---------------------------------------------------------
# Forward final para obtener predicciones
z1 = X @ W1 + b1
a1 = relu(z1)
z2 = a1 @ W2 + b2
y_hat = sigmoid(z2)
pred_clase = (y_hat >= 0.5).astype(int)
acc_final = np.mean(pred_clase == Y)

print("==============================================")
print("RESULTADO FINAL ENTRENAMIENTO")
print("==============================================")
print(f"Pérdida final (loss): {hist_loss[-1]:.4f}")
print(f"Exactitud final     : {acc_final*100:.2f}%")

# Nota:
#   - Esta red ya aprendió una frontera de decisión no lineal.
#   - Lo hizo ajustando pesos con gradiente descendente.
#   - Es básicamente el esqueleto de una red neuronal profunda real.

# ---------------------------------------------------------
# 6. Visualización de la frontera de decisión
# ---------------------------------------------------------
# Vamos a graficar los puntos y la frontera aprendida.
import matplotlib.pyplot as plt

# separar clases verdaderas para graficar
X0 = X[Y[:,0] == 0]
X1 = X[Y[:,0] == 1]

# Creamos una malla (grid) para evaluar la red en el plano
xx, yy = np.meshgrid(
    np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
    np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]  # (200*200, 2)

# forward sobre el grid (para pintar la frontera)
z1_grid = grid_points @ W1 + b1
a1_grid = relu(z1_grid)
z2_grid = a1_grid @ W2 + b2
y_grid = sigmoid(z2_grid)  # prob clase 1
y_grid = y_grid.reshape(xx.shape)

plt.figure(figsize=(6,6))

# Región de decisión: coloreamos según probabilidad
plt.contourf(xx, yy, y_grid >= 0.5, alpha=0.3, levels=[-1,0,1], cmap="bwr")
plt.contour(xx, yy, y_grid, levels=[0.5], colors='black', linewidths=2)

# Puntos reales
plt.scatter(X0[:,0], X0[:,1], c="royalblue", edgecolors="k", label="Clase 0")
plt.scatter(X1[:,0], X1[:,1], c="tomato", edgecolors="k", label="Clase 1")

plt.title("Frontera aprendida por la red neuronal (MLP)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.axis("equal")
plt.show()
