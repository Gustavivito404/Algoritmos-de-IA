# =========================================================
# 36 - RETROPROPAGACIÓN DEL ERROR (BACKPROP)
# ---------------------------------------------------------
# Descripción:
#   En este script entrenamos una red neuronal multicapa
#   (MLP) sobre el problema XOR usando gradiente descendente
#   y retropropagación del error.
#
#   ¿Qué es retropropagación?
#       Es el algoritmo que calcula cómo debe ajustarse cada
#       peso de la red para reducir el error total.
#
#   Flujo por época:
#       1) FORWARD:
#          - Calculamos las salidas capa por capa.
#
#       2) CÁLCULO DE ERROR:
#          - Comparamos la salida de la red con la salida deseada.
#
#       3) BACKWARD (RETROPROP):
#          - Propagamos el error hacia atrás usando derivadas
#            parciales (regla de la cadena).
#
#       4) UPDATE:
#          - Ajustamos los pesos en dirección que disminuye el error.
#
#   En este script además:
#       - Guardamos "fotogramas" de la frontera de decisión
#         en distintos momentos del entrenamiento
#         (época 1, 50, 200, ..., final),
#         para visualizar cómo la red va aprendiendo a separar XOR.
#
# Arquitectura:
#       Entrada (2)
#         ↓
#       Capa oculta (4 neuronas, tanh)
#         ↓
#       Capa salida (1 neurona, sigmoide)
#
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Dataset XOR (no linealmente separable)
# ---------------------------------------------------------
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
], dtype=float)

y = np.array([[0],[1],[1],[0]], dtype=float)

# ---------------------------------------------------------
# 2. Hiperparámetros y estructura de la red
# ---------------------------------------------------------
input_dim  = 2
hidden_dim = 4
output_dim = 1

lr     = 0.1      # learning rate
EPOCHS = 5000     # iteraciones totales

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
    # derivada de la sigmoide con respecto a su entrada neta z:
    s = sigmoide(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x)**2

# ---------------------------------------------------------
# 4. Utilidad: función forward para cualquier entrada
# ---------------------------------------------------------
def forward_pass(X_batch, W1, b1, W2, b2):
    """
    Hace el pase hacia adelante.
    Regresa todos los términos intermedios que necesitamos
    luego en backprop.
    """
    z1 = X_batch @ W1 + b1    # (N, hidden_dim)
    a1 = tanh(z1)             # (N, hidden_dim)

    z2 = a1 @ W2 + b2         # (N, 1)
    a2 = sigmoide(z2)         # (N, 1)

    return z1, a1, z2, a2

# ---------------------------------------------------------
# 5. Para visualizar la frontera de decisión en distintos momentos
# ---------------------------------------------------------
def frontera_decision(W1, b1, W2, b2, titulo, ax):
    """
    Dibuja la frontera de decisión actual (prob > 0.5)
    y los puntos XOR originales.
    """
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]  # (40000, 2)

    # forward en la malla
    z1g = grid @ W1 + b1
    a1g = tanh(z1g)
    z2g = a1g @ W2 + b2
    preds_grid = sigmoide(z2g).reshape(xx.shape)

    # región clasificada
    ax.contourf(xx, yy, preds_grid > 0.5,
                alpha=0.3,
                cmap="coolwarm",
                levels=[-1,0,1])

    # puntos originales XOR
    ax.scatter(X[y[:,0]==0][:,0], X[y[:,0]==0][:,1],
               color="royalblue", edgecolors="k", label="Clase 0")
    ax.scatter(X[y[:,0]==1][:,0], X[y[:,0]==1][:,1],
               color="tomato", edgecolors="k", label="Clase 1")

    ax.set_title(titulo)
    ax.set_xlim(-0.5,1.5)
    ax.set_ylim(-0.5,1.5)
    ax.grid(True, linestyle="--", alpha=0.6)

# ---------------------------------------------------------
# 6. Entrenamiento con retropropagación
#    Guardamos "fotogramas" de la frontera cada cierto tiempo
# ---------------------------------------------------------
snap_epochs = [1, 50, 200, 1000, 3000, EPOCHS-1]  # épocas a guardar
snapshots = {}  # época -> (W1,b1,W2,b2)
loss_history = []

for epoch in range(EPOCHS):
    # --------- FORWARD ---------
    z1, a1, z2, a2 = forward_pass(X, W1, b1, W2, b2)

    # Error cuadrático medio (MSE)
    error = y - a2
    loss = np.mean(error**2)
    loss_history.append(loss)

    # --------- BACKWARD (RETROPROP) ---------
    # dL/da2 = -2*(y - a2)
    d_a2 = -2 * error                            # (4,1)
    # da2/dz2 = d_sigmoide(z2)
    d_z2 = d_a2 * d_sigmoide(z2)                 # (4,1)

    # Gradientes de la capa de salida
    d_W2 = a1.T @ d_z2                           # (hidden_dim,4)@(4,1)->(hidden_dim,1)
    d_b2 = np.sum(d_z2, axis=0, keepdims=True)   # (1,1)

    # Propagar error hacia la capa oculta:
    # dL/da1 = d_z2 @ W2^T
    d_a1 = d_z2 @ W2.T                           # (4,1)@(1,hidden_dim)->(4,hidden_dim)

    # da1/dz1 = d_tanh(z1)
    d_z1 = d_a1 * d_tanh(z1)                     # (4,hidden_dim)

    # Gradientes de la primera capa
    d_W1 = X.T @ d_z1                            # (2,4) = (2,4)
    d_b1 = np.sum(d_z1, axis=0, keepdims=True)   # (1,hidden_dim)

    # --------- UPDATE PESOS ---------
    W2 -= lr * d_W2
    b2 -= lr * d_b2
    W1 -= lr * d_W1
    b1 -= lr * d_b1

    # --------- SNAPSHOT (guardar estado de la red) ---------
    if epoch in snap_epochs:
        snapshots[epoch] = (
            W1.copy(), b1.copy(),
            W2.copy(), b2.copy(),
            loss
        )

# ---------------------------------------------------------
# 7. Resultados finales numéricos
# ---------------------------------------------------------
print("==============================================")
print("RESULTADOS FINALES RETROPROPAGACIÓN")
print("==============================================")

# Forward final
_, _, _, a2_final = forward_pass(X, W1, b1, W2, b2)
pred_final = (a2_final > 0.5).astype(int)

for i in range(len(X)):
    print(f"Entrada {X[i]} -> predicción {pred_final[i,0]} (real {y[i,0]})")

print(f"\nPérdida final (MSE): {loss_history[-1]:.6f}")

# ---------------------------------------------------------
# 8. Visualización de la evolución de la frontera
# ---------------------------------------------------------
fig, axs = plt.subplots(2, 3, figsize=(12,8))
axs = axs.ravel()

for i, ep in enumerate(snap_epochs):
    W1_ep, b1_ep, W2_ep, b2_ep, loss_ep = snapshots[ep]
    frontera_decision(W1_ep, b1_ep, W2_ep, b2_ep,
                      titulo=f"Época {ep} | loss={loss_ep:.3f}",
                      ax=axs[i])

plt.suptitle("Evolución de la frontera de decisión durante el entrenamiento (XOR)", fontsize=14)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 9. Curva de pérdida global (MSE vs época)
# ---------------------------------------------------------
plt.figure(figsize=(7,3))
plt.plot(loss_history, color="purple")
plt.title("Curva de pérdida (MSE) durante el entrenamiento")
plt.xlabel("Época")
plt.ylabel("MSE")
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# Nota:
#   - Observa cómo la frontera empieza siendo caótica,
#     luego se curva y termina separando las esquinas
#     como requiere XOR.
#   - Esto es literalmente ver aprender a una red neuronal.
