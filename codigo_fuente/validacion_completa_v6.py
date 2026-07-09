"""
Validación rigurosa del clasificador de referencia v6 (modo dual):
1. K-fold estratificado (5 folds) para estimar varianza del Recall+
   con más robustez que un solo leave-one-out.
2. Curva ROC/AUC específica de v6 (score continuo, K=15 vecinos),
   la métrica que faltaba desde que se creó v6.
"""
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codigo_fuente.deepwave_preprocessing import calcular_espectrograma_stub
from codigo_fuente.deepwave_features_v2 import extraer_features_v2

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FS_REAL = 2048


def extraer_pico_y_energia_media(espectrograma):
    full = extraer_features_v2(espectrograma)
    return np.array([full[1], full[4]])


def cargar_dataset_v6():
    with open(os.path.join(DATA_DIR, "eventos_procesados.json")) as f:
        registro = json.load(f)
    eventos_todos = registro["eventos_procesados"]
    excluidos = ["GW190620_030421-v2", "GW190630_185205-v2", "GW190708_232457-v2"]
    eventos_con_hl = [e for e in eventos_todos if e not in excluidos]

    positivos = np.load(os.path.join(DATA_DIR, "dataset_real_positivos.npy"))
    negativos = np.load(os.path.join(DATA_DIR, "dataset_real_negativos.npy"))

    indices_con_hl = [i for i, e in enumerate(eventos_todos) if e in eventos_con_hl]
    positivos_sub = positivos[indices_con_hl]
    indices_neg = []
    for i in indices_con_hl:
        indices_neg.extend([2 * i, 2 * i + 1])
    negativos_sub = negativos[indices_neg]

    corr_pos = np.load(os.path.join(DATA_DIR, "correlacion_hl_positivos.npy"))
    corr_neg = np.load(os.path.join(DATA_DIR, "correlacion_hl_negativos.npy"))

    X_pos_sel = np.array([extraer_pico_y_energia_media(calcular_espectrograma_stub(s, FS_REAL)) for s in positivos_sub])
    X_neg_sel = np.array([extraer_pico_y_energia_media(calcular_espectrograma_stub(s, FS_REAL)) for s in negativos_sub])

    X_pos = np.hstack([X_pos_sel, corr_pos.reshape(-1, 1)])
    X_neg = np.hstack([X_neg_sel, corr_neg.reshape(-1, 1)])
    return X_pos, X_neg


def knn_predecir(X_train, y_train, features_test, k):
    X_min, X_max = X_train.min(axis=0), X_train.max(axis=0)
    X_norm = (X_train - X_min) / (X_max - X_min + 1e-10)
    f_norm = (features_test - X_min) / (X_max - X_min + 1e-10)
    distancias = np.sum((X_norm - f_norm) ** 2, axis=1)
    k_cercanos = np.argsort(distancias)[:k]
    conteo = np.bincount(y_train[k_cercanos], minlength=2)
    return np.argmax(conteo)


def score_continuo(X_train, y_train, features_test, k):
    X_min, X_max = X_train.min(axis=0), X_train.max(axis=0)
    X_norm = (X_train - X_min) / (X_max - X_min + 1e-10)
    f_norm = (features_test - X_min) / (X_max - X_min + 1e-10)
    distancias = np.sum((X_norm - f_norm) ** 2, axis=1)
    k_cercanos = np.argsort(distancias)[:k]
    return np.mean(y_train[k_cercanos] == 1)


def kfold_estratificado(X_pos, X_neg, k_vecinos=1, n_folds=5, semilla=42):
    """K-fold manual (sin sklearn) estratificado: cada fold mantiene
    la proporción de positivos/negativos del conjunto completo."""
    rng = np.random.RandomState(semilla)
    idx_pos = rng.permutation(len(X_pos))
    idx_neg = rng.permutation(len(X_neg))

    folds_pos = np.array_split(idx_pos, n_folds)
    folds_neg = np.array_split(idx_neg, n_folds)

    recalls_pos, recalls_neg, globales = [], [], []

    for fold in range(n_folds):
        test_idx_pos = folds_pos[fold]
        test_idx_neg = folds_neg[fold]
        train_idx_pos = np.concatenate([folds_pos[i] for i in range(n_folds) if i != fold])
        train_idx_neg = np.concatenate([folds_neg[i] for i in range(n_folds) if i != fold])

        X_train = np.vstack([X_pos[train_idx_pos], X_neg[train_idx_neg]])
        y_train = np.array([1] * len(train_idx_pos) + [0] * len(train_idx_neg))

        aciertos_pos = sum(knn_predecir(X_train, y_train, X_pos[i], k_vecinos) == 1 for i in test_idx_pos)
        aciertos_neg = sum(knn_predecir(X_train, y_train, X_neg[i], k_vecinos) == 0 for i in test_idx_neg)

        rp = aciertos_pos / len(test_idx_pos) if len(test_idx_pos) > 0 else 0
        rn = aciertos_neg / len(test_idx_neg) if len(test_idx_neg) > 0 else 0
        g = (aciertos_pos + aciertos_neg) / (len(test_idx_pos) + len(test_idx_neg))

        recalls_pos.append(rp)
        recalls_neg.append(rn)
        globales.append(g)
        print(f"  Fold {fold+1}/{n_folds}: Global={g:.1%}, Recall+={rp:.1%}, Recall-={rn:.1%}")

    return np.array(globales), np.array(recalls_pos), np.array(recalls_neg)


def curva_roc_v6(X_pos, X_neg, k_score=15):
    X_todo = np.vstack([X_pos, X_neg])
    y_todo = np.array([1] * len(X_pos) + [0] * len(X_neg))

    scores = []
    for i in range(len(X_todo)):
        X_train = np.delete(X_todo, i, axis=0)
        y_train = np.delete(y_todo, i)
        scores.append(score_continuo(X_train, y_train, X_todo[i], k_score))

    umbrales = sorted(set(scores), reverse=True)
    umbrales = [1.1] + umbrales + [-0.1]
    puntos = []
    total_pos, total_neg = sum(y_todo), len(y_todo) - sum(y_todo)

    for u in umbrales:
        preds = [1 if s >= u else 0 for s in scores]
        tp = sum(1 for p, t in zip(preds, y_todo) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(preds, y_todo) if p == 1 and t == 0)
        tpr = tp / total_pos if total_pos > 0 else 0
        fpr = fp / total_neg if total_neg > 0 else 0
        puntos.append((fpr, tpr))

    auc = 0.0
    for i in range(1, len(puntos)):
        fpr_prev, tpr_prev = puntos[i - 1]
        fpr_curr, tpr_curr = puntos[i]
        auc += (fpr_curr - fpr_prev) * (tpr_curr + tpr_prev) / 2

    return puntos, auc


if __name__ == "__main__":
    print("🌌 VALIDACIÓN COMPLETA DE v6 (K-fold + ROC/AUC)")
    print("=" * 65)

    X_pos, X_neg = cargar_dataset_v6()
    print(f"Dataset: {len(X_pos)} positivos, {len(X_neg)} negativos\n")

    print("🔬 K-FOLD ESTRATIFICADO (5 folds, K=1 vecino)")
    print("-" * 65)
    globales, recalls_pos, recalls_neg = kfold_estratificado(X_pos, X_neg, k_vecinos=1, n_folds=5)
    print(f"\n📊 Global:   media={globales.mean():.1%}, std={globales.std():.1%}")
    print(f"📊 Recall+:  media={recalls_pos.mean():.1%}, std={recalls_pos.std():.1%}")
    print(f"📊 Recall-:  media={recalls_neg.mean():.1%}, std={recalls_neg.std():.1%}")

    print("\n🔬 CURVA ROC/AUC (leave-one-out, score continuo K=15)")
    print("-" * 65)
    puntos, auc = curva_roc_v6(X_pos, X_neg)
    print(f"AUC (v6) = {auc:.3f}")
    print(f"AUC (v1, referencia previa) = 0.754")
    print(f"\n{'FPR':>8} {'TPR':>8}")
    for fpr, tpr in puntos[::2]:
        print(f"{fpr:>8.3f} {tpr:>8.3f}")

    np.save(os.path.join(DATA_DIR, "roc_puntos_v6.npy"), np.array(puntos))
