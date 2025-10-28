# =========================================================
# 42 - RECUPERACIÓN DE DATOS (TF-IDF + SIMILITUD DEL COSENO)
# ---------------------------------------------------------
# Descripción:
#   En este ejemplo construimos un mini "motor de búsqueda"
#   probabilístico clásico basado en:
#       - Frecuencia de términos (TF)
#       - Frecuencia inversa de documento (IDF)
#       - Vectorización TF-IDF
#       - Similitud del coseno
#
#   Objetivo:
#       Dada una consulta, encontrar los documentos más
#       relevantes del corpus según la similitud estadística
#       entre sus vectores TF-IDF.
#
#   Este es el fundamento de los sistemas de
#   "Recuperación de Información" (IR), base de los
#   motores de búsqueda y del paso de “retrieval”
#   en los modelos RAG modernos.
# =========================================================

import numpy as np
from collections import Counter
from typing import List, Dict

# ---------------------------------------------------------
# 1. Corpus de documentos
# ---------------------------------------------------------
documentos = [
    "la niña come comida en la mesa",
    "el niño juega con la pelota en el parque",
    "la inteligencia artificial estudia patrones de datos",
    "la red neuronal aprende representaciones útiles",
    "el sistema de recuperación de información busca textos relevantes",
    "la niña juega en el parque con el niño",
]

# Consulta del usuario
consulta = "recuperación de información en sistemas inteligentes"

# ---------------------------------------------------------
# 2. Preprocesamiento: tokenización simple
# ---------------------------------------------------------
def tokenizar(texto: str) -> List[str]:
    """Convierte el texto a minúsculas y lo separa por espacios."""
    return texto.lower().split()

docs_tokens = [tokenizar(d) for d in documentos]
consulta_tokens = tokenizar(consulta)

# ---------------------------------------------------------
# 3. Construcción del vocabulario global
# ---------------------------------------------------------
vocab = sorted(list(set([t for doc in docs_tokens for t in doc])))
indice_vocab = {pal: idx for idx, pal in enumerate(vocab)}

# ---------------------------------------------------------
# 4. TF (Term Frequency)
# ---------------------------------------------------------
def tf_vector(tokens: List[str], indice_vocab: Dict[str, int]) -> np.ndarray:
    """Calcula la frecuencia normalizada de cada palabra en un documento."""
    conteo = Counter(tokens)
    vec = np.zeros(len(indice_vocab), dtype=float)
    total = len(tokens)
    for palabra, c in conteo.items():
        if palabra in indice_vocab:
            vec[indice_vocab[palabra]] = c / total
    return vec

tf_docs = np.vstack([tf_vector(doc, indice_vocab) for doc in docs_tokens])
tf_q = tf_vector(consulta_tokens, indice_vocab)

# ---------------------------------------------------------
# 5. IDF (Inverse Document Frequency)
# ---------------------------------------------------------
N = len(documentos)
df = np.zeros(len(vocab), dtype=float)

for j, palabra in enumerate(vocab):
    for doc in docs_tokens:
        if palabra in doc:
            df[j] += 1

idf = np.log(N / (1 + df))

# ---------------------------------------------------------
# 6. TF-IDF = TF * IDF
# ---------------------------------------------------------
tfidf_docs = tf_docs * idf
tfidf_q = tf_q * idf

# ---------------------------------------------------------
# 7. Similitud del coseno
# ---------------------------------------------------------
def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Calcula la similitud del coseno entre dos vectores."""
    num = np.dot(a, b)
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return num / den

# ---------------------------------------------------------
# 8. Ranking de documentos por similitud con la consulta
# ---------------------------------------------------------
scores = []
for i, vec_doc in enumerate(tfidf_docs):
    sim = cos_sim(tfidf_q, vec_doc)
    scores.append((i, sim))

scores_ordenados = sorted(scores, key=lambda x: x[1], reverse=True)

# ---------------------------------------------------------
# 9. Resultados
# ---------------------------------------------------------
print("==============================================")
print("RECUPERACIÓN DE DATOS (TF-IDF + coseno)")
print("==============================================\n")

print("Consulta:")
print(f"  {consulta}\n")
print("Ranking de documentos más relevantes:\n")

for rank, (idx_doc, score) in enumerate(scores_ordenados):
    print(f"#{rank+1}  Score={score:.4f}")
    print(f"    Documento[{idx_doc}]: {documentos[idx_doc]}")
    print("")

# ---------------------------------------------------------
# 10. Interpretación
# ---------------------------------------------------------
"""
Interpretación:
---------------
- Cada documento se convierte en un vector TF-IDF.
- La consulta también se convierte en un vector TF-IDF.
- Medimos la similitud coseno entre la consulta y cada documento.
- Los documentos con mayor similitud son los más relevantes.

Este modelo fue la base de los buscadores clásicos,
y aún hoy se usa como paso de "retrieval" previo
a modelos más avanzados como los basados en embeddings
(BERT, GPT, etc.).
"""
