"""
Mide la correlación individual de cada feature (de v1 y v2 combinadas)
con la etiqueta real (BBH=1, ruido=0), sobre el dataset de 40 eventos.
Objetivo: saber cuáles features aportan señal real antes de seguir
añadiendo más a ciegas.
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codigo_fuente.deepwave_preprocessing import calcular_espectrograma_stub
from codigo_fuente.deepwave_features_v2 import extraer_features_v2

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FS_REAL = 2048

NOMBRES_FEATURES = [
    "energia_baja", "energia_media", "energia_alta", "pendiente",
    "pico_max", "varianza_total", "entropia_espectral", "num_picos"
]

def correlacion_point_biserial(feature_values, etiquetas):
    """Correlación entre una variable continua (feature) y una binaria
    (etiqueta 0/1) — el equivalente de Pearson para este caso."""
    return np.corrcoef(feature_values, etiquetas)[0, 1]

if __name__ == "__main__":
    print("🔬 IMPORTANCIA DE FEATURES: correlación individual con la etiqueta real")
    print("=" * 75)

    positivos = np.load(os.path.join(DATA_DIR, "dataset_real_positivos.npy"))
    negativos = np.load(os.path.join(DATA_DIR, "dataset_real_negativos.npy"))

    X, y = [], []
    for s in positivos:
        X.append(extraer_features_v2(calcular_espectrograma_stub(s, FS_REAL)))
        y.append(1)
    for s in negativos:
        X.append(extraer_features_v2(calcular_espectrograma_stub(s, FS_REAL)))
        y.append(0)

    X = np.array(X)
    y = np.array(y)

    print(f"Dataset: {len(y)} muestras ({sum(y)} BBH, {len(y)-sum(y)} ruido)\n")
    print(f"{'Feature':<20} {'Correlación':>12}   Interpretación")
    print("-" * 75)

    resultados = []
    for i, nombre in enumerate(NOMBRES_FEATURES):
        corr = correlacion_point_biserial(X[:, i], y)
        resultados.append((abs(corr), nombre, corr))

    resultados.sort(reverse=True)
    for abs_corr, nombre, corr in resultados:
        if abs_corr > 0.3:
            interpretacion = "🟢 útil"
        elif abs_corr > 0.15:
            interpretacion = "🟡 débil"
        else:
            interpretacion = "🔴 ruido"
        print(f"{nombre:<20} {corr:>+12.3f}   {interpretacion}")

    print("\n📋 |correlación| > 0.3 = relación notable; < 0.15 = prácticamente ruido")
    print("   (con solo 40 eventos positivos, cualquier correlación es ruidosa,")
    print("    pero sirve para descartar candidatas claramente inútiles)")
