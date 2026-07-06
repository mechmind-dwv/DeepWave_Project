"""
Experimento honesto: prueba distintas configuraciones (K, balanceo de
clases) sobre el MISMO dataset real, con leave-one-out, para ver cuál
generaliza mejor. Sin trampas: cada configuración se evalúa igual.
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
    """Pre-calcula features una sola vez (evita recomputar STFT miles de veces)."""
    positivos = np.load(os.path.join(DATA_DIR, "dataset_real_positivos.npy"))
    negativos = np.load(os.path.join(DATA_DIR, "dataset_real_negativos.npy"))

    features_pos = [extraer_features(calcular_espectrograma_stub(s, FS_REAL)) for s in positivos]
    features_neg = [extraer_features(calcular_espectrograma_stub(s, FS_REAL)) for s in negativos]
    return np.array(features_pos), np.array(features_neg)

def knn_predecir(X_train, y_train, features_test, k):
    X_min, X_max = X_train.min(axis=0), X_train.max(axis=0)
    X_norm = (X_train - X_min) / (X_max - X_min + 1e-10)
    f_norm = (features_test - X_min) / (X_max - X_min + 1e-10)
    distancias = np.sum((X_norm - f_norm) ** 2, axis=1)
    k_cercanos = np.argsort(distancias)[:k]
    conteo = np.bincount(y_train[k_cercanos], minlength=2)
    prediccion = np.argmax(conteo)
    confianza = conteo[prediccion] / k
    return prediccion, confianza

def leave_one_out(X_pos, X_neg, k, balancear, semilla=42):
    """balancear=True usa solo tantos negativos como positivos hay,
    elegidos aleatoriamente con semilla fija (reproducible)."""
    rng = np.random.RandomState(semilla)

    if balancear:
        indices_neg = rng.choice(len(X_neg), size=len(X_pos), replace=False)
        X_neg_usado = X_neg[indices_neg]
    else:
        X_neg_usado = X_neg

    X_todo = np.vstack([X_pos, X_neg_usado])
    y_todo = np.array([1]*len(X_pos) + [0]*len(X_neg_usado))

    aciertos_pos, aciertos_neg = 0, 0
    total_pos, total_neg = len(X_pos), len(X_neg_usado)

    for i in range(len(X_todo)):
        X_train = np.delete(X_todo, i, axis=0)
        y_train = np.delete(y_todo, i)
        pred, _ = knn_predecir(X_train, y_train, X_todo[i], k)
        correcto = (pred == y_todo[i])
        if y_todo[i] == 1:
            aciertos_pos += correcto
        else:
            aciertos_neg += correcto

    precision_global = (aciertos_pos + aciertos_neg) / len(X_todo)
    recall_pos = aciertos_pos / total_pos
    recall_neg = aciertos_neg / total_neg
    return precision_global, recall_pos, recall_neg, total_pos, total_neg

if __name__ == "__main__":
    print("🔬 EXPERIMENTO: K-NN real, distintas configuraciones (leave-one-out)")
    print("=" * 75)

    X_pos, X_neg = cargar_features_reales()
    print(f"Dataset: {len(X_pos)} positivos, {len(X_neg)} negativos reales\n")

    configuraciones = [
        (1, False, "K=1, sin balancear (dataset completo)"),
        (3, False, "K=3, sin balancear (dataset completo) [línea base]"),
        (1, True,  "K=1, balanceado (1:1)"),
        (3, True,  "K=3, balanceado (1:1)"),
        (5, True,  "K=5, balanceado (1:1)"),
    ]

    print(f"{'Configuración':<45} {'Global':>8} {'Recall+':>9} {'Recall-':>9}")
    print("-" * 75)
    for k, balancear, nombre in configuraciones:
        precision, recall_pos, recall_neg, n_pos, n_neg = leave_one_out(X_pos, X_neg, k, balancear)
        print(f"{nombre:<45} {precision:>7.1%} {recall_pos:>8.1%} {recall_neg:>8.1%}")

    print("\n📋 Recall+ = % de eventos reales detectados correctamente (lo que más importa)")
    print("   Recall- = % de ruido correctamente rechazado")
