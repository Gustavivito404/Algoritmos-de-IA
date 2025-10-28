# =========================================================
# 29 - MÁQUINAS DE VECTORES DE SOPORTE (SVM con Núcleo RBF)
# ---------------------------------------------------------
# Descripción:
#   Una Máquina de Vectores de Soporte (SVM) busca encontrar
#   una frontera de decisión que maximice el margen entre
#   dos clases de datos.
#
#   Concepto base:
#     - Los puntos más cercanos a la frontera se llaman
#       "vectores de soporte".
#     - La SVM busca un hiperplano que los separe con el
#       mayor margen posible.
#
#   Núcleo RBF:
#     - Permite crear fronteras curvadas al mapear los
#       datos a un espacio de mayor dimensión.
#     - Ideal para datos no linealmente separables.
#
#   Este script:
#     1) Genera datos de dos clases no lineales.
#     2) Entrena una SVM con kernel RBF.
#     3) Visualiza la frontera de decisión y vectores soporte.
#
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# ---------------------------------------------------------
# 1. Generación de datos no linealmente separables
# ---------------------------------------------------------
np.random.seed(42)

# Clase 1: puntos en forma de círculo interior
r_inner = 0.8
theta_inner = np.linspace(0, 2*np.pi, 50)
x_inner = np.c_[r_inner*np.cos(theta_inner), r_inner*np.sin(theta_inner)] + np.random.normal(0, 0.1, (50, 2))

# Clase 2: círculo exterior
r_outer = 1.8
theta_outer = np.linspace(0, 2*np.pi, 50)
x_outer = np.c_[r_outer*np.cos(theta_outer), r_outer*np.sin(theta_outer)] + np.random.normal(0, 0.1, (50, 2))

X = np.vstack((x_inner, x_outer))
y = np.hstack((np.zeros(len(x_inner)), np.ones(len(x_outer))))  # 0 = clase interna, 1 = externa

# ---------------------------------------------------------
# 2. Entrenamiento de la SVM con núcleo RBF
# ---------------------------------------------------------
modelo = svm.SVC(kernel='rbf', gamma='auto', C=1.0)
modelo.fit(X, y)

# ---------------------------------------------------------
# 3. Visualización de la frontera de decisión
# ---------------------------------------------------------
plt.figure(figsize=(7,7))

# Malla para visualizar la frontera
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

# Predicción sobre la malla
Z = modelo.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Contorno de decisión
plt.contourf(xx, yy, Z > 0, alpha=0.3, cmap='bwr')
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

# Dibujar los puntos
plt.scatter(X[y==0,0], X[y==0,1], c='royalblue', label='Clase 0 (interna)', edgecolors='k')
plt.scatter(X[y==1,0], X[y==1,1], c='tomato', label='Clase 1 (externa)', edgecolors='k')

# Vectores de soporte
plt.scatter(modelo.support_vectors_[:, 0],
            modelo.support_vectors_[:, 1],
            s=150, facecolors='none', edgecolors='black',
            linewidths=1.5, label='Vectores de soporte')

plt.title("SVM con Núcleo RBF (Frontera de Decisión)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.axis("equal")
plt.show()

# ---------------------------------------------------------
# 4. Resultados y trazas
# ---------------------------------------------------------
print("==============================================")
print("TRACE SVM (entrenamiento y frontera)")
print("==============================================")
print(f"Número total de puntos: {len(X)}")
print(f"Vectores de soporte: {len(modelo.support_vectors_)}")
print(f"Coeficientes del modelo (dual coef_): {modelo.dual_coef_.shape}")
print(f"Intersección (bias b): {modelo.intercept_}")
print("==============================================")
print("Interpretación:")
print("- Los puntos con borde negro son vectores de soporte.")
print("- La línea negra es la frontera de decisión (Z=0).")
print("- RBF (Radial Basis Function) permite que la frontera")
print("  se curve, separando incluso clases circulares.")
