"""
Aplica el mismo protocolo de validación (k-fold + ROC/AUC) que se usó
para v6, pero a v1 (solo H1, 3 features originales), sobre el dataset
COMPLETO de 75 eventos — no el subconjunto de 68 con H1+L1.
Comparación metodológicamente justa: mismo método, mismo tamaño
máximo de datos disponible para cada versión.
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codigo_fuente.deepwave_preprocessing import calcular_espectrograma_stub
from codigo_fuente.deepwave_knn_real import extraer_features
from codigo_fuente.validacion_completa_v6 import knn_predecir, score_continuo, kfold_estratificado, curva_roc_v6

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FS_REAL = 2048

if __name__ == "__main__":
    print("🌌 VALIDACIÓN COMPLETA DE v1 (K-fold + ROC/AUC) — dataset de 75 eventos")
    print("=" * 75)

    positivos = np.load(os.path.join(DATA_DIR, "dataset_real_positivos.npy"))
    negativos = np.load(os.path.join(DATA_DIR, "dataset_real_negativos.npy"))

    X_pos = np.array([extraer_features(calcular_espectrograma_stub(s, FS_REAL)) for s in positivos])
    X_neg = np.array([extraer_features(calcular_espectrograma_stub(s, FS_REAL)) for s in negativos])

    print(f"Dataset: {len(X_pos)} positivos, {len(X_neg)} negativos\n")

    print("🔬 K-FOLD ESTRATIFICADO (5 folds, K=1 vecino)")
    print("-" * 75)
    globales, recalls_pos, recalls_neg = kfold_estratificado(X_pos, X_neg, k_vecinos=1, n_folds=5)
    print(f"\n📊 Global:   media={globales.mean():.1%}, std={globales.std():.1%}")
    print(f"📊 Recall+:  media={recalls_pos.mean():.1%}, std={recalls_pos.std():.1%}")
    print(f"📊 Recall-:  media={recalls_neg.mean():.1%}, std={recalls_neg.std():.1%}")

    print("\n🔬 CURVA ROC/AUC (leave-one-out, score continuo K=15)")
    print("-" * 75)
    puntos, auc = curva_roc_v6(X_pos, X_neg)  # función genérica, funciona para cualquier X_pos/X_neg
    print(f"AUC (v1, 75 eventos) = {auc:.3f}")
    print(f"AUC (v6, 68 eventos con H1+L1) = 0.708")
    print(f"AUC (v1, 40 eventos, sesión anterior) = 0.754")

    np.save(os.path.join(DATA_DIR, "roc_puntos_v1_75eventos.npy"), np.array(puntos))
