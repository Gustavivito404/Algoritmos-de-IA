# =========================================================
# 30 - APRENDIZAJE PROFUNDO (VERSIÓN KERAS)
# ---------------------------------------------------------
# Descripción:
#   Entrenamos una red neuronal feedforward (MLP) con
#   TensorFlow / Keras para clasificar puntos 2D en dos clases.
#
#   Comparación con la versión NumPy:
#     - Antes hicimos forward/backprop/updates a mano.
#     - Ahora solo definimos arquitectura y Keras
#       se encarga del entrenamiento.
#
#   Arquitectura usada:
#       Capa densa 1: 8 neuronas, activación ReLU
#       Capa densa 2: 4 neuronas, activación ReLU
#       Capa salida : 1 neurona, activación sigmoide
#
#   Optimización:
#       - Binary cross entropy
#       - Adam (descenso de gradiente adaptativo)
#
#   Al final:
#       - Imprimimos accuracy.
#       - Dibujamos la frontera de decisión aprendida.
#
#   Nota:
#       Este patrón (Dense + ReLU + Sigmoid) es literal
#       la base de clasificadores MLP reales en la práctica.
#
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------------------
# 1. Generación de datos (igual idea que en la versión NumPy)
# ---------------------------------------------------------
np.random.seed(42)

N0 = 50
N1 = 50

# Clase 0 cerca de (0,0)
clase0 = np.random.normal(loc=[0.0, 0.0], scale=0.5, size=(N0, 2))
y0 = np.zeros((N0, 1))

# Clase 1 cerca de (2,2)
clase1 = np.random.normal(loc=[2.0, 2.0], scale=0.5, size=(N1, 2))
y1 = np.ones((N1, 1))

# Juntamos todo
X = np.vstack([clase0, clase1])  # shape (100,2)
Y = np.vstack([y0, y1])          # shape (100,1)

# Mezclamos (shuffle consistente)
idx = np.arange(len(X))
np.random.shuffle(idx)
X = X[idx]
Y = Y[idx]

# ---------------------------------------------------------
# 2. Definimos el modelo de red neuronal en Keras
# ---------------------------------------------------------
model = Sequential([
    Dense(8, activation="relu", input_shape=(2,)),  # capa oculta 1 (8 neuronas)
    Dense(4, activation="relu"),                    # capa oculta 2 (4 neuronas)
    Dense(1, activation="sigmoid")                  # salida binaria (probabilidad clase 1)
])

# Compilamos el modelo:
#   - loss: qué tan mal estamos (binaria porque hay 2 clases)
#   - optimizer: Adam (aprendizaje adaptativo)
#   - metrics: accuracy para monitorear desempeño
model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(learning_rate=0.05),
    metrics=["accuracy"]
)

# ---------------------------------------------------------
# 3. Entrenamiento
# ---------------------------------------------------------
print("==============================================")
print("TRACE ENTRENAMIENTO KERAS")
print("==============================================")

hist = model.fit(
    X, Y,
    epochs=200,          # épocas de entrenamiento
    batch_size=16,       # mini-lotes
    verbose=0            # 0 = silencioso, si quieres ver cada época pon 1
)

# Imprimimos algunas épocas de referencia
for mark in [0, 50, 100, 150, 199]:
    loss_mark = hist.history["loss"][mark]
    acc_mark = hist.history["accuracy"][mark]
    print(f"Época {mark:3d} | loss={loss_mark:.4f} | acc={acc_mark*100:.2f}%")

# ---------------------------------------------------------
# 4. Evaluación final
# ---------------------------------------------------------
loss_final, acc_final = model.evaluate(X, Y, verbose=0)

print("\n==============================================")
print("RESULTADO FINAL ENTRENAMIENTO (KERAS)")
print("==============================================")
print(f"Loss final      : {loss_final:.4f}")
print(f"Accuracy final  : {acc_final*100:.2f}%")

# ---------------------------------------------------------
# 5. Frontera de decisión aprendida
# ---------------------------------------------------------
# Creamos una malla 2D y pedimos predicciones al modelo
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

grid_points = np.c_[xx.ravel(), yy.ravel()]  # (40000, 2)
probs = model.predict(grid_points, verbose=0)  # prob clase 1
probs = probs.reshape(xx.shape)

# ---------------------------------------------------------
# 6. Visualización
# ---------------------------------------------------------
plt.figure(figsize=(6,6))

# 6a. Región clasificada
plt.contourf(xx, yy, probs >= 0.5,
             alpha=0.3,
             levels=[-1,0,1],
             cmap="bwr")

# 6b. Línea de frontera (prob=0.5)
plt.contour(xx, yy, probs,
            levels=[0.5],
            colors='black',
            linewidths=2)

# 6c. Puntos reales con clase verdadera
X0 = X[Y[:,0] == 0]
X1 = X[Y[:,0] == 1]

plt.scatter(X0[:,0], X0[:,1],
            c="royalblue", edgecolors="k",
            label="Clase 0")
plt.scatter(X1[:,0], X1[:,1],
            c="tomato", edgecolors="k",
            label="Clase 1")

plt.title("Frontera aprendida por la red neuronal (Keras)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.axis("equal")
plt.show()

# ---------------------------------------------------------
# 7. Comentario final
# ---------------------------------------------------------
print("\nNota:")
print("- Esta red es ya una red neuronal 'real' con múltiples capas.")
print("- Adam actualiza pesos automáticamente con backprop interno.")
print("- La frontera negra en la gráfica es dónde el modelo cambia")
print("  de predecir Clase 0 a Clase 1 (probabilidad = 0.5).")
print("- Así es como se construyen clasificadores en visión, voz, etc.")
