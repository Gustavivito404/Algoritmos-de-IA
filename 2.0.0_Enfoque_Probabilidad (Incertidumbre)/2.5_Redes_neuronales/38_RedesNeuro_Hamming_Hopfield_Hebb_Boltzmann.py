# =========================================================
# 38 - REDES RECURRENTES CLÁSICAS: HOPFIELD, HEBB, ETC.
# ---------------------------------------------------------
# Descripción:
#   En este script implementamos una Red de Hopfield,
#   que es un tipo de red neuronal recurrente que funciona
#   como una MEMORIA ASOCIATIVA.
#
#   Idea:
#     - Tú le enseñas patrones binarios (por ejemplo imágenes
#       de 1 y -1).
#     - La red "almacena" esos patrones en su matriz de pesos.
#     - Luego le das una versión RUIDOSA o INCOMPLETA
#       de uno de esos patrones...
#     - ...y la red intenta recuperar el patrón original
#       recordado más cercano.
#
#   Esto se parece a "corregir memoria borrosa".
#
#   Matemática básica (Hopfield):
#     - Estados de neuronas: s_i ∈ {+1, -1}
#     - Matriz de pesos simétrica: W (sin autoconexiones)
#     - Dinámica asíncrona:
#           s_i ← sign( Σ_j W_ij * s_j )
#
#     - La red converge hacia *atractores estables*
#       que son, idealmente, los patrones aprendidos.
#
#   Conexión con Hebb:
#     - Los pesos se aprenden usando una forma de la
#       regla de Hebb:
#
#           W_ij = Σ_p ( s_i^p * s_j^p ),  i ≠ j
#
#       Es decir: "neuronas que se activan juntas,
#       se conectan más fuerte".
#
#   En este script:
#     1) Definimos 2 patrones binarios 5x5 (por ejemplo letras).
#     2) Entrenamos la red Hopfield con ellos.
#     3) Le damos una versión dañada de uno de los patrones.
#     4) Observamos si la red lo recupera.
#
#   Además:
#     - Al final del script dejamos una explicación corta
#       de: Hebb, Hamming, Hopfield, Boltzmann.
# =========================================================

import numpy as np

# ---------------------------------------------------------
# 1. Utilidades para mostrar patrones en consola
# ---------------------------------------------------------
def mostrar_patron(vec, shape=(5,5), titulo=""):
    """Imprime un patrón 2D bonito usando '#' y '.'"""
    if titulo:
        print(titulo)
    grid = vec.reshape(shape)
    for fila in grid:
        linea = "".join("#" if v == 1 else "." for v in fila)
        print(linea)
    print("")

# ---------------------------------------------------------
# 2. Definimos patrones que queremos memorizar
# ---------------------------------------------------------
# Vamos a usar +1 y -1 en lugar de 1 y 0.
# Ejemplo: dos letras muy simples en una cuadrícula 5x5.

# Letra "A"
patron_A = np.array([
    [-1,  1,  1,  1, -1],
    [ 1, -1, -1, -1,  1],
    [ 1,  1,  1,  1,  1],
    [ 1, -1, -1, -1,  1],
    [ 1, -1, -1, -1,  1],
], dtype=int)

# Letra "T"
patron_T = np.array([
    [ 1,  1,  1,  1,  1],
    [-1, -1, 1, -1, -1],
    [-1, -1, 1, -1, -1],
    [-1, -1, 1, -1, -1],
    [-1, -1, 1, -1, -1],
], dtype=int)

# Aplanamos (vectorizamos) cada patrón 5x5 -> 25x1
pA = patron_A.reshape(-1, 1)  # (25,1)
pT = patron_T.reshape(-1, 1)  # (25,1)

# Conjunto de patrones a memorizar
patrones = [pA, pT]

print("==============================================")
print("PATRONES ORIGINALES (Memorias deseadas)")
print("==============================================")
mostrar_patron(pA, titulo="Patrón A (objetivo a memorizar):")
mostrar_patron(pT, titulo="Patrón T (objetivo a memorizar):")


# ---------------------------------------------------------
# 3. Entrenamiento de la red de Hopfield
# ---------------------------------------------------------
def entrenar_hopfield(patrones):
    """
    Regla de Hebb generalizada:
        W = sum( p * p^T ) sobre todos los patrones p
    con ceros en la diagonal (sin auto-conexión).
    """
    n = patrones[0].shape[0]   # número de neuronas (25)
    W = np.zeros((n, n))

    for p in patrones:
        W += p @ p.T  # suma outer product

    # Eliminamos auto-pesos:
    np.fill_diagonal(W, 0)

    return W

W = entrenar_hopfield(patrones)

print("==============================================")
print("MATRIZ DE PESOS ENTRENADA (W)")
print("==============================================")
print(W)
print("")


# ---------------------------------------------------------
# 4. Dinámica de actualización (recuperación de memoria)
# ---------------------------------------------------------
def signo(x):
    # Signo binario: +1 si x>=0, -1 si x<0
    return np.where(x >= 0, 1, -1)

def actualizar_hopfield_estado(s, W, pasos=10, traza=False):
    """
    Dado un estado inicial s (vector de +1/-1),
    aplicamos la dinámica de Hopfield por varios pasos.

    En cada paso:
      - elegimos neuronas una por una (actualización asíncrona)
      - cada neurona i se actualiza usando:
            s_i = sign( Σ_j W_ij * s_j )

    Retorna:
      s_final
    """
    s = s.copy()

    N = s.shape[0]

    for step in range(pasos):
        # Actualizamos neuronas en orden aleatorio
        idxs = np.random.permutation(N)
        for i in idxs:
            net_i = np.dot(W[i, :], s.flatten())
            s[i, 0] = 1 if net_i >= 0 else -1

        if traza:
            print(f"[Paso {step+1}] Estado parcial:")
            mostrar_patron(s, titulo=f"Iteración {step+1}")

    return s

# ---------------------------------------------------------
# 5. Probamos la red con una versión dañada de 'A'
# ---------------------------------------------------------
ruido = pA.copy()
# Volteamos el signo de algunos pixeles para simular ruido
flip_indices = np.random.choice(len(ruido), size=5, replace=False)
ruido[flip_indices] *= -1

print("==============================================")
print("PATRÓN RUIDOSO (entrada a la red)")
print("==============================================")
mostrar_patron(ruido, titulo="Patrón tipo 'A' pero con ruido:")

# Recuperamos usando dinámica de Hopfield
recuperado = actualizar_hopfield_estado(ruido, W, pasos=8, traza=True)

print("==============================================")
print("RESULTADO FINAL DE RECUPERACIÓN")
print("==============================================")
mostrar_patron(recuperado, titulo="Patrón recuperado por la red:")

# ---------------------------------------------------------
# 6. ¿A qué memoria se parece más el recuperado?
# ---------------------------------------------------------
def distancia_hamming(vec1, vec2):
    """
    Distancia de Hamming entre dos vectores binarios (+1/-1):
    cuenta cuántos bits difieren.
    """
    return np.sum(vec1.flatten() != vec2.flatten())

dist_A = distancia_hamming(recuperado, pA)
dist_T = distancia_hamming(recuperado, pT)

print("==============================================")
print("ANÁLISIS DE IDENTIDAD (Distancia de Hamming)")
print("==============================================")
print(f"Distancia al patrón A: {dist_A}")
print(f"Distancia al patrón T: {dist_T}")

if dist_A < dist_T:
    print("→ La red cree que esto era una 'A'")
elif dist_T < dist_A:
    print("→ La red cree que esto era una 'T'")
else:
    print("→ Ambiguo / punto intermedio entre memorias")


# ---------------------------------------------------------
# 7. Resumen teórico rápido
# ---------------------------------------------------------
# Nota:
#   A continuación incluyo las ideas para estudiar.

"""
-----------------------------------------------------------
HEBB (Regla de Hebb)
-----------------------------------------------------------
"Neurons that fire together wire together."
Traducción: Si dos neuronas se activan juntas (+1/+1),
reforzamos su conexión positiva. Si una está en +1 y la otra
en -1, debilitamos o hacemos negativa la conexión.

En forma simple:
    ΔW_ij ∝ s_i * s_j

Esto construye una memoria distribuida: el patrón queda
codificado en los pesos.

-----------------------------------------------------------
HOPFIELD
-----------------------------------------------------------
- Red recurrente y simétrica (W_ij = W_ji).
- Estado de cada neurona es binario (+1 o -1).
- Los patrones almacenados son "atractores".
- Si das una versión incompleta/ruidosa, el sistema evoluciona
  paso a paso hasta caer en el atractor más cercano.
- Se usa mucho como modelo de memoria asociativa.

En este script implementamos EXACTAMENTE eso.

-----------------------------------------------------------
DISTANCIA DE HAMMING
-----------------------------------------------------------
- Mide cuántas posiciones difieren entre dos patrones.
- La usamos al final para ver si el patrón recuperado por
  Hopfield es más parecido a 'A' o a 'T'.

-----------------------------------------------------------
RED HAMMING
-----------------------------------------------------------
- Una red Hamming clásica clasifica un patrón nuevo midiendo
  la similitud con prototipos almacenados usando distancia
  de Hamming / correlación.
- Es más tipo "clasificador directo", no dinámica recurrente.

-----------------------------------------------------------
MÁQUINA DE BOLTZMANN
-----------------------------------------------------------
- Parecida a Hopfield pero probabilística / estocástica.
- Cada neurona se enciende con cierta probabilidad que depende
  de la energía del sistema (tipo física estadística).
- Puede aprender pesos mediante un proceso parecido a
  máxima probabilidad (similar a modelos de energía).
- Es el ancestro conceptual de las Restricted Boltzmann
  Machines usadas en Deep Learning temprano.

-----------------------------------------------------------
CONCLUSIÓN MECATRÓNICA:
-----------------------------------------------------------
- Hebb = cómo se forman memorias en los pesos.
- Hopfield = cómo recuperas memorias estables.
- Hamming = cómo decides cuál memoria es más parecida.
- Boltzmann = cómo exploras estados con probabilidad.

Este archivo te da una memoria asociativa real
que "recuerda" letras dañadas. Muy útil como base
para visión clásica, reconocimiento de patrones,
y teoría de control inteligente con memoria.
"""
