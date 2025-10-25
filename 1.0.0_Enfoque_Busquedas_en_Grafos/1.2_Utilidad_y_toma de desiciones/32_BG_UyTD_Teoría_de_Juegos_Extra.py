# =========================================================
# 32b - TEORÍA DE JUEGOS: MECANISMO DE SUBASTA (VICKREY)
# ---------------------------------------------------------
# Descripción:
#   Simulación de una subasta de segundo precio (Vickrey Auction).
#
#   Jugadores envían sus ofertas (bids) según su valoración del objeto.
#   El mayor postor gana, pero paga el precio de la segunda oferta más alta.
#
#   En equilibrio de Nash, la estrategia dominante es:
#       -> ofertar tu valor real.
#
# =========================================================

import random

# ---------------------------------------------------------
# 1. Definición de jugadores y sus valores reales (privados)
# ---------------------------------------------------------
jugadores = ["A", "B", "C", "D"]
valores_reales = {j: random.randint(30, 100) for j in jugadores}

# ---------------------------------------------------------
# 2. Estrategia: cada jugador hace una oferta
# ---------------------------------------------------------
# Podemos simular tres comportamientos:
#   - Veraz: oferta = valor real
#   - Conservador: oferta < valor real
#   - Agresivo: oferta > valor real
#
# Aquí haremos mezcla para ver efectos.
def ofertar(valor_real, modo="veraz"):
    if modo == "veraz":
        return valor_real
    if modo == "conservador":
        return int(valor_real * random.uniform(0.7, 0.9))
    if modo == "agresivo":
        return int(valor_real * random.uniform(1.1, 1.3))
    return valor_real

estrategias = {
    "A": "veraz",
    "B": "agresivo",
    "C": "conservador",
    "D": "veraz"
}

ofertas = {j: ofertar(valores_reales[j], estrategias[j]) for j in jugadores}

# ---------------------------------------------------------
# 3. Determinar ganador y precio de pago
# ---------------------------------------------------------
ordenadas = sorted(ofertas.items(), key=lambda x: x[1], reverse=True)
ganador, oferta_ganadora = ordenadas[0]
precio_a_pagar = ordenadas[1][1]  # segunda oferta más alta

# ---------------------------------------------------------
# 4. Calcular utilidades
# ---------------------------------------------------------
utilidades = {}
for j in jugadores:
    if j == ganador:
        utilidades[j] = valores_reales[j] - precio_a_pagar
    else:
        utilidades[j] = 0

# ---------------------------------------------------------
# 5. Mostrar resultados
# ---------------------------------------------------------
print("==============================================")
print("TRACE MECANISMO DE SUBASTA DE VICKREY")
print("==============================================\n")

print("VALORES REALES (privados):")
for j, v in valores_reales.items():
    print(f"  {j}: {v}")

print("\nOFERTAS REALIZADAS:")
for j, o in ofertas.items():
    print(f"  {j}: {o} ({estrategias[j]})")

print("\nRESULTADO:")
print(f"  Ganador: {ganador}")
print(f"  Oferta ganadora: {oferta_ganadora}")
print(f"  Precio a pagar: {precio_a_pagar}")
print("\nUTILIDADES FINALES:")
for j, u in utilidades.items():
    print(f"  {j}: {u}")

# Nota:
# - En equilibrio, ofertar tu valor real ("veraz") es óptimo.
# - Si eres agresivo y excedes tu valor real, puedes perder utilidad.
# - Si eres conservador, puedes perder la subasta aunque te convenía ganar.
# - Este tipo de mecanismo (verdad dominante) se usa en:
#     • subastas de anuncios online (Google Ads)
#     • asignación de recursos multiagente
#     • diseño de sistemas económicos justos y eficientes.
