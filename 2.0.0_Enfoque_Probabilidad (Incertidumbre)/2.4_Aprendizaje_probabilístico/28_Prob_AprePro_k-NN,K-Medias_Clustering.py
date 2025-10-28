# =========================================================
# 28 - k-MEANS, CLUSTERING Y k-NN (COMPARACIÓN DIRECTA)
# ---------------------------------------------------------
# Descripción:
#   En este script comparamos dos enfoques clásicos:
#
#   (1) CLUSTERING NO SUPERVISADO (k-Means)
#       - Sin etiquetas.
#       - Agrupa puntos calculando centroides.
#       - Objetivo: minimizar distancia interna al cluster.
#
#   (2) CLASIFICACIÓN SUPERVISADA (k-NN)
#       - Usa etiquetas conocidas ("Clase A", "Clase B").
#       - Clasifica un punto nuevo viendo a sus k vecinos
#         más cercanos y votando la clase.
#
#   IDEA CLAVE:
#       - k-Means intenta "descubrir" estructura en los datos
#         (aprendizaje no supervisado).
#
#       - k-NN intenta "asignar clase" a un punto nuevo
#         (aprendizaje supervisado).
#
#   Flujo del script:
#       1) Generamos 2 nubes de puntos en 2D (A y B).
#       2) Ejecutamos k-Means con k=2 y mostramos centroides.
#       3) Ejecutamos k-NN sobre un punto nuevo y mostramos
#          los vecinos más cercanos y su voto.
#       4) Graficamos ambos resultados.
#
# =========================================================

from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ---------------------------------------------------------
# 1. Generación de datos
# ---------------------------------------------------------
np.random.seed(7)

# Creamos dos grupos "reales" con etiqueta
clase_A = np.random.normal(loc=[1.0, 1.0], scale=0.4, size=(20, 2))  # nube alrededor de (1,1)
clase_B = np.random.normal(loc=[4.0, 4.0], scale=0.4, size=(20, 2))  # nube alrededor de (4,4)

# Datos combinados (para k-Means, que NO ve etiquetas)
datos_todos = np.vstack((clase_A, clase_B))

# Datos etiquetados (para k-NN)
datos_etiquetados: List[Tuple[np.ndarray, str]] = []
for p in clase_A:
    datos_etiquetados.append((p, "A"))
for p in clase_B:
    datos_etiquetados.append((p, "B"))

# Punto nuevo a clasificar con k-NN
p_nuevo = np.array([2.5, 2.5])

# ---------------------------------------------------------
# 2. Implementación de k-Means (clustering no supervisado)
# ---------------------------------------------------------
def inicializar_centroides(datos: np.ndarray, k: int) -> np.ndarray:
    """Elige k puntos aleatorios del dataset como centroides iniciales."""
    indices = np.random.choice(range(len(datos)), k, replace=False)
    return datos[indices]

def asignar_clusters(datos: np.ndarray, centroides: np.ndarray) -> np.ndarray:
    """Para cada punto, encuentra el centroide más cercano (distancia Euclidiana)."""
    distancias = np.linalg.norm(datos[:, np.newaxis] - centroides, axis=2)  # shape: (n_puntos, k)
    return np.argmin(distancias, axis=1)  # etiqueta de cluster para cada punto

def recalcular_centroides(datos: np.ndarray, etiquetas_cluster: np.ndarray, k: int) -> np.ndarray:
    """Recalcula cada centroide como la media de los puntos que le pertenecen."""
    nuevos = np.array([
        datos[etiquetas_cluster == i].mean(axis=0)
        for i in range(k)
    ])
    return nuevos

def k_means(datos: np.ndarray, k: int, max_iter: int = 10):
    """Ejecuta k-means y devuelve las etiquetas finales y los centroides."""
    centroides = inicializar_centroides(datos, k)

    for it in range(max_iter):
        etiquetas_cluster = asignar_clusters(datos, centroides)
        nuevos_centroides = recalcular_centroides(datos, etiquetas_cluster, k)

        print(f"[k-Means] Iteración {it+1}")
        print(f"Centroides estimados:\n{nuevos_centroides}\n")

        if np.allclose(centroides, nuevos_centroides):
            print("[k-Means] Convergencia alcanzada ✅\n")
            break

        centroides = nuevos_centroides

    return etiquetas_cluster, centroides

# ---------------------------------------------------------
# 3. Implementación de k-NN (clasificación supervisada)
# ---------------------------------------------------------
def distancia_euclidiana(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))

def vecinos_mas_cercanos(
    datos: List[Tuple[np.ndarray, str]],
    punto: np.ndarray,
    k: int
) -> List[Tuple[float, np.ndarray, str]]:
    """
    Regresa los k vecinos más cercanos al punto nuevo:
    lista de tuplas (distancia, coordenada, etiqueta).
    """
    distancias = []
    for (coord, etiqueta) in datos:
        d = distancia_euclidiana(coord, punto)
        distancias.append((d, coord, etiqueta))
    distancias.sort(key=lambda x: x[0])
    return distancias[:k]

def votar(vecinos: List[Tuple[float, np.ndarray, str]]) -> Tuple[str, Dict[str, int]]:
    """Votación mayoritaria entre las clases de los vecinos."""
    etiquetas = [et for (_, _, et) in vecinos]
    conteo = Counter(etiquetas)
    clase_ganadora = conteo.most_common(1)[0][0]
    return clase_ganadora, dict(conteo)

# ---------------------------------------------------------
# 4. Ejecutar k-Means en los datos SIN etiquetas
# ---------------------------------------------------------
k_clusters = 2
etiquetas_clustering, centroides_finales = k_means(datos_todos, k_clusters, max_iter=10)

# ---------------------------------------------------------
# 5. Ejecutar k-NN para clasificar p_nuevo USANDO etiquetas
# ---------------------------------------------------------
k_vecinos = 3
vecinos = vecinos_mas_cercanos(datos_etiquetados, p_nuevo, k_vecinos)
clase_predicha, conteo_votos = votar(vecinos)

# ---------------------------------------------------------
# 6. TRAZAS DE RESULTADOS
# ---------------------------------------------------------
print("==============================================")
print("TRACE k-NN (clasificación supervisada)")
print("==============================================")
print(f"Punto nuevo a clasificar: {p_nuevo}")
print(f"Usando k = {k_vecinos}\n")

print("Vecinos más cercanos:")
for i, (dist, coord, etiqueta) in enumerate(vecinos):
    print(f"  Vecino {i+1}: punto={coord}, clase={etiqueta}, dist={dist:.3f}")

print("\nConteo de votos por clase (k-NN):")
for clase, cnt in conteo_votos.items():
    print(f"  Clase {clase}: {cnt} voto(s)")

print(f"\nCLASE FINAL (k-NN) PARA {p_nuevo}  -->  {clase_predicha}\n")

print("==============================================")
print("TRACE k-Means (clustering no supervisado)")
print("==============================================")
print("Asignación de cluster para los primeros 5 puntos:")
for idx in range(5):
    print(f"  Punto {idx:02d}: {datos_todos[idx]} pertenece al cluster {etiquetas_clustering[idx]}")

print("\nCentroides finales estimados por k-Means:")
print(centroides_finales)

# ---------------------------------------------------------
# 7. GRÁFICAS
#    (a) Resultado de k-Means sobre TODOS los datos
#    (b) Vecinos usados por k-NN
# ---------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12,5))

# -------- Gráfica (a): k-Means clustering --------
axs[0].set_title("k-Means (clustering no supervisado)")
colores_cluster = ['royalblue', 'tomato']
for i in range(k_clusters):
    axs[0].scatter(
        datos_todos[etiquetas_clustering == i, 0],
        datos_todos[etiquetas_clustering == i, 1],
        c=colores_cluster[i],
        alpha=0.6,
        label=f"Cluster {i+1}"
    )
axs[0].scatter(
    centroides_finales[:,0],
    centroides_finales[:,1],
    c='black',
    marker='X',
    s=200,
    label='Centroides'
)
axs[0].set_xlabel("X1")
axs[0].set_ylabel("X2")
axs[0].legend()
axs[0].grid(True, linestyle="--", alpha=0.5)
axs[0].set_aspect("equal")

# -------- Gráfica (b): k-NN vecinos más cercanos --------
axs[1].set_title(f"k-NN (k={k_vecinos}) clasificación supervisada")

# separo por clase real A/B (etiquetas verdaderas)
clase_A_xy = np.array([p for (p, et) in datos_etiquetados if et == "A"])
clase_B_xy = np.array([p for (p, et) in datos_etiquetados if et == "B"])

axs[1].scatter(clase_A_xy[:,0], clase_A_xy[:,1],
               c="royalblue", alpha=0.6, label="Clase A (real)")
axs[1].scatter(clase_B_xy[:,0], clase_B_xy[:,1],
               c="tomato", alpha=0.6, label="Clase B (real)")

# punto nuevo
axs[1].scatter(p_nuevo[0], p_nuevo[1],
               c="black", marker="X", s=200, label="Punto nuevo")

# dibujamos las líneas al punto nuevo desde sus vecinos
for (dist, coord, etiqueta) in vecinos:
    axs[1].plot([p_nuevo[0], coord[0]],
                [p_nuevo[1], coord[1]],
                "--", color="gray", alpha=0.6)
    axs[1].scatter(coord[0], coord[1],
                   edgecolors="black",
                   facecolors="none",
                   s=200,
                   linewidths=1.5)

axs[1].set_xlabel("X1")
axs[1].set_ylabel("X2")
axs[1].legend()
axs[1].grid(True, linestyle="--", alpha=0.5)
axs[1].set_aspect("equal")

plt.suptitle("Comparación: k-Means (no supervisado) vs k-NN (supervisado)")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 8. RESUMEN FINAL
# ---------------------------------------------------------
print("==============================================")
print("RESUMEN FINAL DEL SCRIPT")
print("==============================================")
print("1) k-Means (no supervisado):")
print("   - No usa etiquetas reales A/B.")
print("   - Solo agrupa por cercanía geométrica.")
print("   - Devuelve centroides y asigna cada punto a un cluster.")

print("\n2) k-NN (supervisado):")
print("   - Sí usa las etiquetas A/B dadas por humanos.")
print("   - Para un punto NUEVO, mira los vecinos más cercanos")
print("     y vota la clase más común.")
print(f"   - Clase predicha para {p_nuevo} => {clase_predicha}")

# Nota:
#   k-Means intenta descubrir estructura interna de los datos.
#   k-NN intenta etiquetar un punto desconocido usando ejemplos con etiqueta.
#   Esto resume la diferencia entre aprendizaje NO supervisado vs supervisado.
