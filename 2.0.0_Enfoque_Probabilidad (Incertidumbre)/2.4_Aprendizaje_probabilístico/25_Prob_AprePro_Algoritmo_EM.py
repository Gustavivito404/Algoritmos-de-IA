# =========================================================
# 25 - ALGORITMO EM (Expectation-Maximization)
# ---------------------------------------------------------
# Descripción:
#   El algoritmo EM se usa para estimar parámetros en modelos
#   donde existen variables ocultas (no observadas).
#
#   Ejemplo clásico:
#       Supón que tienes datos provenientes de dos grupos
#       (por ejemplo, alturas de hombres y mujeres),
#       pero no sabes qué punto pertenece a qué grupo.
#
#   EM alterna dos fases:
#       E-step (Esperanza):  Calcula la probabilidad de
#                            pertenecer a cada grupo.
#       M-step (Maximización): Actualiza los parámetros
#                              (media, varianza, peso) según
#                              esas probabilidades.
#
# =========================================================

import numpy as np
import matplotlib.pyplot as plt

def normal_pdf(x, mu, sigma):
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma)**2)


# ---------------------------------------------------------
# 1. Generamos datos sin etiquetas (mezcla de gaussianas)
# ---------------------------------------------------------
np.random.seed(42)

# Dos grupos "ocultos"
grupo1 = np.random.normal(0, 1, 100)     # media=0, sigma=1
grupo2 = np.random.normal(5, 1.5, 100)   # media=5, sigma=1.5

# Mezclamos los datos (sin saber a qué grupo pertenecen)
datos = np.concatenate([grupo1, grupo2])
np.random.shuffle(datos)

# ---------------------------------------------------------
# 2. Inicialización de parámetros (adivinados)
# ---------------------------------------------------------
mu1, mu2 = np.random.choice(datos, 2)  # medias iniciales
sigma1, sigma2 = 1.0, 1.0              # desviaciones iniciales
pi1, pi2 = 0.5, 0.5                    # pesos iniciales

# ---------------------------------------------------------
# 3. Función del algoritmo EM
# ---------------------------------------------------------
def EM(datos, mu1, mu2, sigma1, sigma2, pi1, pi2, iteraciones=10):
    """Ejecuta el algoritmo EM paso a paso."""
    for i in range(iteraciones):
        # ----- E-step -----
        # Calcular responsabilidades (probabilidad de pertenecer a cada grupo)
        r1 = pi1 * normal_pdf(datos, mu1, sigma1)
        r2 = pi2 * normal_pdf(datos, mu2, sigma2)
        suma = r1 + r2
        r1 /= suma
        r2 /= suma

        # ----- M-step -----
        # Actualizar parámetros usando las responsabilidades
        N1 = np.sum(r1)
        N2 = np.sum(r2)
        mu1 = np.sum(r1 * datos) / N1
        mu2 = np.sum(r2 * datos) / N2
        sigma1 = np.sqrt(np.sum(r1 * (datos - mu1)**2) / N1)
        sigma2 = np.sqrt(np.sum(r2 * (datos - mu2)**2) / N2)
        pi1 = N1 / len(datos)
        pi2 = N2 / len(datos)

        # Mostrar resultados intermedios
        print(f"Iteración {i+1}:")
        print(f"  μ1={mu1:.2f}, σ1={sigma1:.2f}, π1={pi1:.2f}")
        print(f"  μ2={mu2:.2f}, σ2={sigma2:.2f}, π2={pi2:.2f}\n")

    return mu1, mu2, sigma1, sigma2, pi1, pi2, r1, r2

# ---------------------------------------------------------
# 4. Ejecución del algoritmo
# ---------------------------------------------------------
mu1, mu2, sigma1, sigma2, pi1, pi2, r1, r2 = EM(datos, mu1, mu2, sigma1, sigma2, pi1, pi2, iteraciones=10)

# ---------------------------------------------------------
# 5. Resultados finales
# ---------------------------------------------------------
print("==============================================")
print("RESULTADOS FINALES (Algoritmo EM)")
print("==============================================")
print(f"Grupo 1 → μ={mu1:.2f}, σ={sigma1:.2f}, peso={pi1:.2f}")
print(f"Grupo 2 → μ={mu2:.2f}, σ={sigma2:.2f}, peso={pi2:.2f}")
print("\nInterpretación:")
print(" - EM logra separar los dos grupos sin conocer etiquetas.")
print(" - Estima medias, varianzas y proporciones de mezcla.\n")

# ---------------------------------------------------------
# 6. Visualización opcional
# ---------------------------------------------------------
x = np.linspace(min(datos), max(datos), 200)
y1 = pi1 * normal_pdf(x, mu1, sigma1)
y2 = pi2 * normal_pdf(x, mu2, sigma2)

plt.hist(datos, bins=30, density=True, alpha=0.6, color="gray", label="Datos observados")
plt.plot(x, y1, label="Componente 1 (estimado)", color="blue")
plt.plot(x, y2, label="Componente 2 (estimado)", color="red")
plt.title("Algoritmo EM - Separación de Mezcla de Gaussianas")
plt.xlabel("Valor")
plt.ylabel("Densidad de probabilidad")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
