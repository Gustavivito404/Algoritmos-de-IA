# =========================================================
# 37 - MAPAS AUTOORGANIZADOS DE KOHONEN (SOM)
# ---------------------------------------------------------
# Descripción:
#   Entrenamos un SOM (Self-Organizing Map) 2D de 10x10
#   usando colores RGB como datos de entrada.
#
#   Este SOM aprende SIN supervisión:
#     - No le decimos la clase correcta.
#     - Solo agrupa "colores parecidos" en neuronas vecinas.
#
#   En este script guardamos "fotogramas" del mapa en:
#       - época   0% (inicial)
#       - época  25%
#       - época  50%
#       - época 100% (final)
#
#   Así visualizamos cómo el mapa pasa de caos aleatorio
#   → a un gradiente suave de color.
#
#   Fórmula de actualización de los pesos w_ij:
#
#       w_ij(t+1) = w_ij(t)
#                   + α(t) * h_bmu(i,j,t) * (x - w_ij(t))
#
#   donde:
#       α(t)               = tasa de aprendizaje decreciente
#       h_bmu(i,j,t)       = vecindad gaussiana alrededor de la BMU
#       x                  = vector de entrada aleatorio (ej. [R,G,B])
#       BMU (Best Matching Unit) = neurona más parecida a x
#
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Generación del conjunto de datos (colores RGB)
# ---------------------------------------------------------
# Creamos 500 colores aleatorios en espacio RGB normalizado [0,1]
datos = np.random.rand(500, 3)

# ---------------------------------------------------------
# 2. Definición de la red SOM
# ---------------------------------------------------------
class SOM:
    def __init__(self, m, n, dim, lr=0.5, sigma=None, epochs=100):
        """
        m, n    -> tamaño del mapa (m filas x n columnas)
        dim     -> dimensión de cada vector de entrada (RGB -> 3)
        lr      -> learning rate inicial
        sigma   -> radio de vecindad inicial
        epochs  -> épocas totales de entrenamiento
        """
        self.m, self.n, self.dim = m, n, dim
        self.lr = lr
        self.sigma = sigma if sigma else max(m, n) / 2
        self.epochs = epochs

        # Inicializamos pesos aleatorios en [0,1]
        self.weights = np.random.rand(m, n, dim)

    def _distancia(self, x):
        """
        Distancia euclidiana entre x (1xdim) y cada neurona del mapa.
        Regresa matriz (m x n) con la distancia a cada neurona.
        """
        return np.linalg.norm(self.weights - x, axis=2)

    def _bmu(self, x):
        """
        Encuentra la BMU (Best Matching Unit)
        = la neurona más parecida al vector x.
        Regresa índices (i, j) en la grilla.
        """
        distancias = self._distancia(x)
        return np.unravel_index(np.argmin(distancias), (self.m, self.n))

    def entrenar_con_snapshots(self, datos, snapshot_epochs):
        """
        Entrena actualizando los pesos paso a paso.
        Además, guarda "fotos" (copias de self.weights)
        en las épocas listadas en snapshot_epochs.
        
        Return:
            snapshots: dict {epoch_int: pesos_en_ese_momento}
        """
        snapshots = {}
        for t in range(self.epochs):
            # Decaimiento progresivo del learning rate y del radio de vecindad
            lr_t = self.lr * np.exp(-t / self.epochs)
            sigma_t = self.sigma * np.exp(-t / (self.epochs / np.log(self.sigma)))

            # Selecciona muestra aleatoria del dataset
            x = datos[np.random.randint(0, len(datos))]

            # Encuentra la neurona ganadora (BMU)
            bmu = self._bmu(x)

            # Actualiza TODAS las neuronas según su distancia a la BMU
            for i in range(self.m):
                for j in range(self.n):
                    # Distancia en el mapa (coordenadas 2D del grid)
                    dist_ij = np.linalg.norm(
                        np.array([i, j]) - np.array(bmu)
                    )

                    # h_bmu(i,j,t): vecindad gaussiana
                    h = np.exp(-(dist_ij**2) / (2 * (sigma_t**2)))

                    # Regla de actualización hacia x
                    self.weights[i, j, :] += lr_t * h * (x - self.weights[i, j, :])

            # Guardar snapshot si esta época nos importa
            if t in snapshot_epochs:
                snapshots[t] = self.weights.copy()

            # Traza básica
            if (t+1) % (self.epochs // 5) == 0:
                print(f"[Época {t+1}/{self.epochs}] lr={lr_t:.4f} sigma={sigma_t:.4f}")

        # Aseguramos también el snapshot final
        snapshots[self.epochs-1] = self.weights.copy()
        return snapshots


# ---------------------------------------------------------
# 3. Configuración y entrenamiento con snapshots
# ---------------------------------------------------------
m = 10
n = 10
EPOCHS = 100

som = SOM(m=m, n=n, dim=3, lr=0.5, epochs=EPOCHS)

# Definimos en qué épocas queremos "foto":
snapshot_epochs = [
    0,                          # inicio total
    int(EPOCHS * 0.25),         # 25%
    int(EPOCHS * 0.50),         # 50%
    EPOCHS - 1                  # final (100%)
]

# Guardamos el estado inicial ANTES de entrenar
snapshots_iniciales = {0: som.weights.copy()}

# Entrenamos y guardamos snapshots intermedios
snapshots_entrenamiento = som.entrenar_con_snapshots(datos, snapshot_epochs)

# Unimos ambos (por si epoch 0 ya estaba en snapshot_epochs, igual sobrescribe con lo mismo)
snapshots = {**snapshots_iniciales, **snapshots_entrenamiento}

# ---------------------------------------------------------
# 4. Visualización de la evolución del mapa SOM
# ---------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(16, 4))

titulos = [
    f"Inicio (época 0)",
    f"25% (época {int(EPOCHS*0.25)})",
    f"50% (época {int(EPOCHS*0.50)})",
    f"Final (época {EPOCHS-1})"
]

epocas_a_mostrar = [
    0,
    int(EPOCHS*0.25),
    int(EPOCHS*0.50),
    EPOCHS-1
]

for ax, ep, titulo in zip(axs, epocas_a_mostrar, titulos):
    ax.imshow(snapshots[ep], interpolation="nearest")
    ax.set_title(titulo)
    ax.axis("off")

plt.suptitle("Evolución del SOM (Mapa de Colores RGB)", fontsize=14)
plt.show()

# ---------------------------------------------------------
# 5. Nota final
# ---------------------------------------------------------
# Comentario:
#   - Fíjate cómo el primer panel es puro caos.
#   - A 25% ya empiezan "manchas" de color similar.
#   - A 50% se ven gradientes más suaves.
#   - Al final (100%) el mapa parece casi continuo,
#     como si el SOM hubiera ordenado el espacio RGB
#     en bloques coherentes.
#
#   Esto es autoorganización topológica:
#   neuronas vecinas en la grilla terminan
#   representando colores parecidos.
