# =========================================================
# 34 - SEPARABILIDAD LINEAL
# ---------------------------------------------------------
# Descripción:
#   Demostración visual del concepto de separabilidad lineal.
#
#   Un problema es linealmente separable si existe una
#   **recta (en 2D)** o **hiperplano (en N dimensiones)**
#   que pueda separar completamente las clases.
#
#   Ejemplo separable: AND
#   Ejemplo no separable: XOR
#
#   En este script:
#       - Se entrena un perceptrón clásico sobre los dos casos.
#       - Se muestran las fronteras aprendidas.
#       - Se evidencia por qué el perceptrón no puede aprender XOR.
#
#   Este fue un punto histórico en la IA: el fracaso con XOR
#   motivó la creación de las redes multicapa con funciones
#   de activación no lineales.
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Definimos conjuntos de datos lógicos en 2D
# ---------------------------------------------------------
# AND:
#   (0,0)->0
#   (0,1)->0
#   (1,0)->0
#   (1,1)->1   <-- sólo este es 1
X_AND = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
], dtype=float)
y_AND = np.array([[0],[0],[0],[1]], dtype=float)

# XOR:
#   (0,0)->0
#   (0,1)->1
#   (1,0)->1
#   (1,1)->0   <-- el opuesto de igualdad
X_XOR = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
], dtype=float)
y_XOR = np.array([[0],[1],[1],[0]], dtype=float)

# Agregamos bias como tercera entrada fija en 1
def agregar_bias(X):
    return np.hstack([X, np.ones((X.shape[0],1))])

X_AND_b = agregar_bias(X_AND)  # shape (4,3)
X_XOR_b = agregar_bias(X_XOR)  # shape (4,3)

# ---------------------------------------------------------
# 2. Perceptrón clásico (una neurona con activación escalón)
# ---------------------------------------------------------
def escalon(z):
    return np.where(z >= 0, 1, 0)

def entrenar_perceptron(X, y, lr=0.1, epochs=50):
    """
    Regla clásica del perceptrón:
    w := w + lr * (d - y_pred) * x
    donde:
    - d es la salida deseada (y real)
    - y_pred es salida actual (0/1)
    """
    w = np.random.randn(X.shape[1], 1)

    for ep in range(epochs):
        for i in range(X.shape[0]):
            z = np.dot(X[i], w)
            y_pred = escalon(z)
            error = y[i] - y_pred
            w += lr * error * X[i].reshape(-1,1)
    return w

# Entrenamos perceptrón en AND y XOR
w_and = entrenar_perceptron(X_AND_b, y_AND, lr=0.2, epochs=50)
w_xor = entrenar_perceptron(X_XOR_b, y_XOR, lr=0.2, epochs=50)

# ---------------------------------------------------------
# 3. Función para graficar frontera de decisión en 2D
# ---------------------------------------------------------
def graficar_frontera(ax, X, y, w, titulo):
    """
    Dibuja:
      - puntos clase 0 y 1
      - la frontera lineal aprendida por el perceptrón
    """
    # separar las clases para graficar
    clase0 = X[y[:,0] == 0]
    clase1 = X[y[:,0] == 1]

    ax.scatter(clase0[:,0], clase0[:,1],
               c="royalblue", edgecolors="k", label="Clase 0")
    ax.scatter(clase1[:,0], clase1[:,1],
               c="tomato", edgecolors="k", label="Clase 1")

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(titulo)

    # w tiene 3 términos: w0, w1, b  (donde b es w[2])
    # Ecuación del perceptrón:
    #   z = w0*x + w1*y + b
    # Frontera de decisión en el plano z=0:
    #   w0*x + w1*y + b = 0
    #   => y = (-w0*x - b)/w1   (si w1 != 0)

    w0 = w[0,0]
    w1 = w[1,0]
    b  = w[2,0]

    if abs(w1) > 1e-6:
        xs = np.linspace(-0.5, 1.5, 50)
        ys = (-w0*xs - b) / w1
        ax.plot(xs, ys, 'k-', linewidth=2, label="Frontera perc.")
    else:
        # Frontera casi vertical: w1≈0 → w0*x + b = 0 → x = -b/w0
        x_const = -b / (w0 + 1e-12)
        ax.axvline(x_const, color='k', linewidth=2, label="Frontera perc.")

    ax.legend(loc="upper left", fontsize=8)

# ---------------------------------------------------------
# 4. Visualización lado a lado AND vs XOR
# ---------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10,5))

graficar_frontera(axs[0], X_AND, y_AND, w_and,
                  titulo="AND (Separable)")

graficar_frontera(axs[1], X_XOR, y_XOR, w_xor,
                  titulo="XOR (NO separable)")

plt.suptitle("Perceptrón frente a AND vs XOR", fontsize=14)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 5. Evaluación numérica rápida
# ---------------------------------------------------------
def predecir_perceptron(X, w):
    z = X @ w
    return escalon(z)

y_pred_and = predecir_perceptron(X_AND_b, w_and)
y_pred_xor = predecir_perceptron(X_XOR_b, w_xor)

acc_and = np.mean(y_pred_and == y_AND)
acc_xor = np.mean(y_pred_xor == y_XOR)

print("==============================================")
print("RESULTADOS PERCEPTRÓN")
print("==============================================")
print(f"Exactitud en AND (esperado alta): {acc_and*100:.1f}%")
print(f"Exactitud en XOR (esperado baja): {acc_xor*100:.1f}%")

# Nota:
#   - AND se puede separar con una sola recta.
#   - XOR NO se puede separar con una sola recta.
#   - Aquí ves por qué el perceptrón clásico fracasa con XOR.
#   - Para resolver XOR necesitamos al menos:
#         • 2 neuronas ocultas en paralelo
#         • una capa oculta no lineal
#     ...es decir, una red multicapa con activación no lineal.
