"""
Curva ROC/AUC real sobre el dataset de 40 eventos + 80 negativos,
usando leave-one-out (cada muestra excluida de su propio cálculo de
vecinos). El score continuo es la proporción de votos BBH entre los
K vecinos más cercanos, no solo la clase mayoritaria.
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepwave_preprocessing import calcular_espectrograma_stub
from deepwave_knn_real import extraer_features

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FS_REAL = 2048

def cargar_features_reales():
    positivos = np.load(os.path.join(DATA_DIR, "dataset_real_positivos.npy"))
    negativos = np.load(os.path.join(DATA_DIR, "dataset_real_negativos.npy"))
    features_pos = [extraer_features(calcular_espectrograma_stub(s, FS_REAL)) for s in positivos]
    features_neg = [extraer_features(calcular_espectrograma_stub(s, FS_REAL)) for s in negativos]
    return np.array(features_pos), np.array(features_neg)

def score_continuo(X_train, y_train, features_test, k):
    """Devuelve la proporción de vecinos BBH entre los k más cercanos:
    un score real entre 0.0 y 1.0, no solo la clase mayoritaria."""
    X_min, X_max = X_train.min(axis=0), X_train.max(axis=0)
    X_norm = (X_train - X_min) / (X_max - X_min + 1e-10)
    f_norm = (features_test - X_min) / (X_max - X_min + 1e-10)
    distancias = np.sum((X_norm - f_norm) ** 2, axis=1)
    k_cercanos = np.argsort(distancias)[:k]
    return np.mean(y_train[k_cercanos] == 1)

def calcular_roc_auc_manual(y_true, y_scores):
    """Implementación manual de ROC/AUC (sin sklearn, para no depender
    de otra compilación pesada) — método de Mann-Whitney U."""
    umbrales = sorted(set(y_scores), reverse=True)
    umbrales = [1.1] + umbrales + [-0.1]

    puntos_roc = []
    positivos_totales = sum(y_true)
    negativos_totales = len(y_true) - positivos_totales

    for umbral in umbrales:
        predicciones = [1 if s >= umbral else 0 for s in y_scores]
        tp = sum(1 for p, t in zip(predicciones, y_true) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(predicciones, y_true) if p == 1 and t == 0)
        tpr = tp / positivos_totales if positivos_totales > 0 else 0
        fpr = fp / negativos_totales if negativos_totales > 0 else 0
        puntos_roc.append((fpr, tpr, umbral))

    # AUC por regla del trapecio
    auc = 0.0
    for i in range(1, len(puntos_roc)):
        fpr_prev, tpr_prev, _ = puntos_roc[i-1]
        fpr_curr, tpr_curr, _ = puntos_roc[i]
        auc += (fpr_curr - fpr_prev) * (tpr_curr + tpr_prev) / 2

    return puntos_roc, auc

if __name__ == "__main__":
    print("📈 CURVA ROC/AUC REAL (leave-one-out, score continuo K=15)")
    print("=" * 65)

    X_pos, X_neg = cargar_features_reales()
    X_todo = np.vstack([X_pos, X_neg])
    y_todo = np.array([1]*len(X_pos) + [0]*len(X_neg))

    K_SCORE = 15  # más vecinos = score más granular (0/15, 1/15, ..., 15/15)
    scores = []
    for i in range(len(X_todo)):
        X_train = np.delete(X_todo, i, axis=0)
        y_train = np.delete(y_todo, i)
        s = score_continuo(X_train, y_train, X_todo[i], K_SCORE)
        scores.append(s)

    puntos_roc, auc = calcular_roc_auc_manual(y_todo, scores)

    print(f"\nAUC = {auc:.3f}")
    print("(0.5 = azar puro, 1.0 = clasificador perfecto, tal como en la literatura estándar)")

    print(f"\n{'Umbral':>8} {'FPR':>8} {'TPR':>8}")
    print("-" * 30)
    for fpr, tpr, umbral in puntos_roc[::2]:  # cada 2 puntos para no saturar la pantalla
        print(f"{umbral:>8.3f} {fpr:>8.3f} {tpr:>8.3f}")

    np.save(os.path.join(DATA_DIR, "roc_puntos.npy"), np.array([(f,t) for f,t,_ in puntos_roc]))
    print(f"\n💾 Puntos ROC guardados en data/roc_puntos.npy")
