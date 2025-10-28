# =========================================================
# 26 - AGRUPAMIENTO NO SUPERVISADO (k-MEANS CLUSTERING)
# ---------------------------------------------------------
# Descripción:
#   El algoritmo k-Means divide los datos en K grupos (clusters)
#   sin conocer etiquetas, minimizando la distancia entre los
#   puntos y el centroide de su grupo.
#
#   Fases:
#       1) Inicialización aleatoria de centroides.
#       2) Asignación: cada punto va al centroide más cercano.
#       3) Actualización: se recalcula el centroide del grupo.
#       4) Se repite hasta converger.
#
#   Nota:
#       - Similar al Algoritmo EM, pero determinista (sin
#         probabilidades, solo distancias euclidianas).
#
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Generamos datos de ejemplo (dos grupos)
# ---------------------------------------------------------
np.random.seed(42)
grupo1 = np.random.normal([0, 0], [1, 1], (50, 2))
grupo2 = np.random.normal([5, 5], [1, 1], (50, 2))
datos = np.vstack((grupo1, grupo2))

# ---------------------------------------------------------
# 2. Parámetros iniciales
# ---------------------------------------------------------
k = 2  # número de clusters
centroides = datos[np.random.choice(range(len(datos)), k, replace=False)]

def asignar_clusters(datos, centroides):
    """Asigna cada punto al centroide más cercano."""
    distancias = np.linalg.norm(datos[:, np.newaxis] - centroides, axis=2)
    return np.argmin(distancias, axis=1)

def recalcular_centroides(datos, etiquetas, k):
    """Calcula los nuevos centroides como el promedio de los puntos asignados."""
    nuevos = np.array([datos[etiquetas == i].mean(axis=0) for i in range(k)])
    return nuevos

# ---------------------------------------------------------
# 3. Bucle iterativo del algoritmo
# ---------------------------------------------------------
MAX_ITER = 10

for i in range(MAX_ITER):
    etiquetas = asignar_clusters(datos, centroides)
    nuevos_centroides = recalcular_centroides(datos, etiquetas, k)

    print(f"Iteración {i+1}")
    print(f"Centroides:\n{nuevos_centroides}\n")

    # Si no cambian los centroides, detenemos
    if np.allclose(centroides, nuevos_centroides):
        print("Convergencia alcanzada ✅\n")
        break
    centroides = nuevos_centroides

# ---------------------------------------------------------
# 4. Visualización final
# ---------------------------------------------------------
colores = ['royalblue', 'tomato']
for i in range(k):
    plt.scatter(datos[etiquetas == i, 0], datos[etiquetas == i, 1],
                c=colores[i], label=f"Cluster {i+1}", alpha=0.6)
plt.scatter(centroides[:, 0], centroides[:, 1], c='black', marker='X', s=200, label='Centroides')
plt.title("k-Means Clustering - Agrupamiento no supervisado")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# ---------------------------------------------------------
# 5. Interpretación
# ---------------------------------------------------------
print("==============================================")
print("RESULTADO FINAL")
print("==============================================")
print(" - Cada punto fue asignado al centroide más cercano.")
print(" - El algoritmo converge cuando los centroides dejan de moverse.")
print(" - Es la base de técnicas de segmentación, compresión y clustering.")
