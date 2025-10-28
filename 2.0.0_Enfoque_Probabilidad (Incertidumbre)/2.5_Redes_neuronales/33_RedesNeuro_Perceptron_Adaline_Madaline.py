# =========================================================
# 33 - PERCEPTRÓN, ADALINE y MADALINE
# ---------------------------------------------------------
# Descripción:
#   Comparación de tres modelos neuronales históricos:
#
#   🧠 PERCEPTRÓN (Rosenblatt, 1958)
#       - Clasificador lineal binario.
#       - Usa activación escalón.
#       - Aprende solo si los datos son separables linealmente.
#
#   ⚙️ ADALINE (Widrow & Hoff, 1960)
#       - Usa salida continua.
#       - Actualiza pesos por el gradiente del error cuadrático medio.
#       - Más estable que el perceptrón.
#
#   🔗 MADALINE (Multiple ADALINEs, 1962)
#       - Red multicapa formada por varias ADALINEs.
#       - Antecesora directa del aprendizaje por retropropagación.
#
#   En este script:
#       - Se generan datos 2D binarios.
#       - Se entrena cada modelo.
#       - Se visualizan las fronteras de decisión.
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Generación de datos binarios separables linealmente
# ---------------------------------------------------------
np.random.seed(42)
N = 50

# Clase 0 cerca de (0, 0)
X0 = np.random.normal(loc=[0.2, 0.2], scale=0.3, size=(N, 2))
y0 = np.zeros((N, 1))

# Clase 1 cerca de (1, 1)
X1 = np.random.normal(loc=[1.0, 1.0], scale=0.3, size=(N, 2))
y1 = np.ones((N, 1))

# Mezclamos datos
X = np.vstack([X0, X1])
y = np.vstack([y0, y1])

# Agregamos bias (columna de 1s)
Xb = np.hstack([X, np.ones((X.shape[0], 1))])

# ---------------------------------------------------------
# 2. Funciones auxiliares
# ---------------------------------------------------------
def escalon(z):
    return np.where(z >= 0, 1, 0)

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

# ---------------------------------------------------------
# 3. Perceptrón
# ---------------------------------------------------------
def entrenar_perceptron(X, y, lr=0.1, epochs=20):
    w = np.random.randn(X.shape[1], 1)
    for ep in range(epochs):
        for i in range(X.shape[0]):
            z = np.dot(X[i], w)
            y_pred = escalon(z)
            error = y[i] - y_pred
            w += lr * error * X[i].reshape(-1, 1)
    return w

# ---------------------------------------------------------
# 4. ADALINE (usa salida lineal + MSE)
# ---------------------------------------------------------
def entrenar_adaline(X, y, lr=0.01, epochs=5000):
    w = np.random.randn(X.shape[1], 1)
    for ep in range(epochs):
        y_pred = X @ w
        error = y - y_pred
        w += lr * X.T @ error / len(X)
    return w

# ---------------------------------------------------------
# 5. MADALINE (2 neuronas ADALINE + capa de salida)
# ---------------------------------------------------------
def entrenar_madaline(X, y, lr=1.4, epochs=50):
    hidden_neurons = 2
    W1 = np.random.randn(X.shape[1], hidden_neurons)
    W2 = np.random.randn(hidden_neurons, 1)
    for ep in range(epochs):
        # Forward
        Z1 = X @ W1
        A1 = np.tanh(Z1)
        Z2 = A1 @ W2
        y_pred = sigmoide(Z2)
        # Error y gradientes
        error = y - y_pred
        dW2 = A1.T @ (error * y_pred * (1 - y_pred)) / len(X)
        dA1 = (error * y_pred * (1 - y_pred)) @ W2.T
        dW1 = X.T @ (dA1 * (1 - A1**2)) / len(X)
        # Actualización
        W2 += lr * dW2
        W1 += lr * dW1
    return W1, W2

# ---------------------------------------------------------
# 6. Entrenamiento
# ---------------------------------------------------------
w_perc = entrenar_perceptron(Xb, y)
w_ada = entrenar_adaline(Xb, y)
W1_mad, W2_mad = entrenar_madaline(Xb, y)

# ---------------------------------------------------------
# 7. Visualización de resultados
# ---------------------------------------------------------
x_min, x_max = X[:,0].min()-0.2, X[:,0].max()+0.2
y_min, y_max = X[:,1].min()-0.2, X[:,1].max()+0.2
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Xg = np.c_[xx.ravel(), yy.ravel(), np.ones(xx.ravel().shape)]

# Predicciones de cada modelo
Z_perc = escalon(Xg @ w_perc).reshape(xx.shape)
Z_ada  = sigmoide(Xg @ w_ada).reshape(xx.shape)
A1_mad = np.tanh(Xg @ W1_mad)
Z_mad  = sigmoide(A1_mad @ W2_mad).reshape(xx.shape)

# ---------------------------------------------------------
# 8. Graficamos
# ---------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
titles = ["Perceptrón", "ADALINE", "MADALINE"]
Zs = [Z_perc, Z_ada, Z_mad]

for ax, Z, title in zip(axs, Zs, titles):
    ax.contourf(xx, yy, Z > 0.5, alpha=0.3, cmap="coolwarm")
    ax.scatter(X0[:,0], X0[:,1], c="royalblue", label="Clase 0", edgecolors="k")
    ax.scatter(X1[:,0], X1[:,1], c="tomato", label="Clase 1", edgecolors="k")
    ax.set_title(title)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle="--", alpha=0.5)

axs[0].legend()
plt.suptitle("Comparación: Perceptrón, ADALINE y MADALINE", fontsize=14)
plt.show()

# ---------------------------------------------------------
# 9. Resumen
# ---------------------------------------------------------
print("==============================================")
print("RESUMEN DE MODELOS")
print("==============================================")
print("""
🧠 PERCEPTRÓN:
   - Usa salida binaria (0/1).
   - Aprende solo si los datos son linealmente separables.
   - Muy rápido pero limitado.

⚙️ ADALINE:
   - Usa salida continua y error cuadrático medio.
   - Aprende con gradiente, más estable.
   - Puede converger incluso con ruido leve.

🔗 MADALINE:
   - Combina varias ADALINE en una red multicapa.
   - Usa activaciones no lineales (tanh, sigmoide).
   - Puede resolver problemas no linealmente separables.
""")
