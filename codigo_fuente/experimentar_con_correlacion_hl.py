"""
Prueba si añadir la correlación H1-L1 (débil individualmente) a las
3 features originales de v1 mejora el K-NN, usando solo los 37
eventos que sí tienen esa correlación calculada.
"""
import numpy as np
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codigo_fuente.deepwave_preprocessing import calcular_espectrograma_stub
from codigo_fuente.deepwave_knn_real import extraer_features
from codigo_fuente.experimentar_knn_real import knn_predecir, leave_one_out

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FS_REAL = 2048

if __name__ == "__main__":
    corr_pos = np.load(os.path.join(DATA_DIR, "correlacion_hl_positivos.npy"))
    corr_neg = np.load(os.path.join(DATA_DIR, "correlacion_hl_negativos.npy"))

    print(f"Correlaciones disponibles: {len(corr_pos)} positivos, {len(corr_neg)} negativos")
    print(f"Media evento={corr_pos.mean():.3f}, media ruido={corr_neg.mean():.3f}")

    # Test estadístico simple: Mann-Whitney U (no asume normalidad,
    # apropiado para muestras pequeñas)
    from scipy.stats import mannwhitneyu
    stat, p_valor = mannwhitneyu(corr_pos, corr_neg, alternative='greater')
    print(f"\nTest Mann-Whitney U (H1: eventos > ruido): p-valor = {p_valor:.4f}")
    if p_valor < 0.05:
        print("✅ Diferencia estadísticamente significativa (p<0.05)")
    else:
        print("⚠️  NO alcanza significancia estadística convencional (p>=0.05)")
