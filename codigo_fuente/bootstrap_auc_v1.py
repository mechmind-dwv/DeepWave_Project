"""
Intervalo de confianza bootstrap (95%) para el AUC del clasificador
v1 (baseline), sobre el dataset completo de 75 eventos.

Método: (1) calcular los scores leave-one-out UNA vez (K=15 vecinos,
igual que en validacion_v1_completa.py); (2) remuestrear con
reemplazo los pares (score, etiqueta) B=2000 veces; (3) para cada
remuestreo, calcular el AUC vía el estadístico de Mann-Whitney
(equivalente exacto al AUC por suma de rangos, mucho más rápido que
reconstruir la curva ROC completa en cada iteración); (4) reportar
los percentiles 2.5% y 97.5% como IC 95%.
"""
import numpy as np
from scipy.stats import rankdata
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codigo_fuente.deepwave_preprocessing import calcular_espectrograma_stub
from codigo_fuente.deepwave_knn_real import extraer_features

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FS_REAL = 2048
K_SCORE = 15
N_BOOTSTRAP = 2000
SEMILLA = 42


def score_continuo(X_train, y_train, features_test, k):
    X_min, X_max = X_train.min(axis=0), X_train.max(axis=0)
    X_norm = (X_train - X_min) / (X_max - X_min + 1e-10)
    f_norm = (features_test - X_min) / (X_max - X_min + 1e-10)
    distancias = np.sum((X_norm - f_norm) ** 2, axis=1)
    k_cercanos = np.argsort(distancias)[:k]
    return np.mean(y_train[k_cercanos] == 1)


def auc_mann_whitney(scores, etiquetas):
    """AUC exacto vía suma de rangos (equivalente matemático al AUC
    por trapecio, pero O(n log n) en vez de O(n * n_umbrales) —
    imprescindible para poder repetirlo miles de veces en bootstrap."""
    rangos = rankdata(scores)
    n_pos = np.sum(etiquetas == 1)
    n_neg = np.sum(etiquetas == 0)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    suma_rangos_pos = np.sum(rangos[etiquetas == 1])
    u = suma_rangos_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


if __name__ == "__main__":
    print("🌌 INTERVALO DE CONFIANZA BOOTSTRAP PARA AUC (v1, baseline)")
    print("=" * 70)

    positivos = np.load(os.path.join(DATA_DIR, "dataset_real_positivos.npy"))
    negativos = np.load(os.path.join(DATA_DIR, "dataset_real_negativos.npy"))

    X_pos = np.array([extraer_features(calcular_espectrograma_stub(s, FS_REAL)) for s in positivos])
    X_neg = np.array([extraer_features(calcular_espectrograma_stub(s, FS_REAL)) for s in negativos])

    X_todo = np.vstack([X_pos, X_neg])
    y_todo = np.array([1] * len(X_pos) + [0] * len(X_neg))
    n = len(y_todo)

    print(f"Dataset: {len(X_pos)} positivos, {len(X_neg)} negativos")
    print(f"Calculando scores leave-one-out (K={K_SCORE})...")

    scores = np.zeros(n)
    for i in range(n):
        X_train = np.delete(X_todo, i, axis=0)
        y_train = np.delete(y_todo, i)
        scores[i] = score_continuo(X_train, y_train, X_todo[i], K_SCORE)

    auc_puntual = auc_mann_whitney(scores, y_todo)
    print(f"\nAUC puntual (todo el dataset) = {auc_puntual:.3f}")

    print(f"\nEjecutando bootstrap ({N_BOOTSTRAP} remuestreos)...")
    rng = np.random.RandomState(SEMILLA)
    aucs_bootstrap = []

    for b in range(N_BOOTSTRAP):
        indices = rng.randint(0, n, size=n)
        scores_b = scores[indices]
        y_b = y_todo[indices]
        auc_b = auc_mann_whitney(scores_b, y_b)
        if not np.isnan(auc_b):
            aucs_bootstrap.append(auc_b)

    aucs_bootstrap = np.array(aucs_bootstrap)
    ic_inferior = np.percentile(aucs_bootstrap, 2.5)
    ic_superior = np.percentile(aucs_bootstrap, 97.5)

    print(f"\n📊 RESULTADO FINAL:")
    print(f"   AUC = {auc_puntual:.3f} (IC 95%: {ic_inferior:.3f}–{ic_superior:.3f})")
    print(f"   Media bootstrap: {aucs_bootstrap.mean():.3f}, std: {aucs_bootstrap.std():.3f}")
    print(f"   Remuestreos válidos: {len(aucs_bootstrap)}/{N_BOOTSTRAP}")

    np.save(os.path.join(DATA_DIR, "bootstrap_auc_v1.npy"), aucs_bootstrap)
    print(f"\n💾 Distribución bootstrap guardada en data/bootstrap_auc_v1.npy")
