"""
Cruza los fallos de clasificación con la masa de chirp real de cada
evento (parámetro físico que determina la velocidad de evolución de
la frecuencia de la señal), usando masas oficiales de GWTC-1
(Phys. Rev. X 9, 031040, 2019, Table VI).
"""

# (m1, m2) en masas solares, valores centrales del catálogo oficial
MASAS = {
    "GW150914": (35.6, 30.6),
    "GW151012": (23.3, 13.6),
    "GW151226": (13.7, 7.7),
    "GW170104": (31.0, 20.1),
    "GW170608": (10.9, 7.6),
    "GW170729": (50.6, 34.3),
    "GW170809": (35.2, 23.8),
    "GW170814": (30.7, 25.3),
    "GW170817": (1.46, 1.27),  # BNS, no BBH
    "GW170818": (35.5, 26.8),
    "GW170823": (39.6, 29.4),
}

# Resultado observado en el diagnóstico (K=3, el más estable)
RESULTADO = {
    "GW150914": "✅ detectado",
    "GW151012": "✅ detectado",
    "GW151226": "✅ detectado (K=1,3)",
    "GW170104": "❌ fallado",
    "GW170608": "❌ fallado",
    "GW170729": "❌ fallado",
    "GW170809": "❌ fallado",
    "GW170814": "✅ detectado",
    "GW170817": "✅ detectado (K=1,3)",
    "GW170818": "✅ detectado",
    "GW170823": "❌ fallado",
}

def masa_chirp(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2

if __name__ == "__main__":
    print("🔬 MASA DE CHIRP vs RESULTADO DE CLASIFICACIÓN")
    print("=" * 60)

    filas = []
    for evento, (m1, m2) in MASAS.items():
        mc = masa_chirp(m1, m2)
        filas.append((mc, evento, RESULTADO[evento]))

    filas.sort()
    print(f"{'Evento':<12} {'Masa chirp (M☉)':>16}   Resultado")
    print("-" * 60)
    for mc, evento, resultado in filas:
        print(f"{evento:<12} {mc:>16.2f}   {resultado}")

    detectados = [mc for mc, e, r in filas if "✅" in r]
    fallados = [mc for mc, e, r in filas if "❌" in r]
    print(f"\nMasa chirp promedio - detectados: {sum(detectados)/len(detectados):.2f} M☉")
    print(f"Masa chirp promedio - fallados:   {sum(fallados)/len(fallados):.2f} M☉")
