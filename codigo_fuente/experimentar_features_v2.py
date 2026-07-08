"""
Compara el K-NN con features v1 (3 estadísticos) contra v2 (8
estadísticos), sobre el MISMO dataset real de 40 eventos, con
leave-one-out. Determina si más features rompe la meseta de Recall+
que se identificó con v1, o si el cuello de botella está en otro lado.
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codigo_fuente.deepwave_preprocessing import calcular_espectrograma_stub
from codigo_fuente.deepwave_knn_real import extraer_features as extraer_features_v1
from codigo_fuente.deepwave_features_v2 import extraer_features_v2

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FS_REAL = 2048

def cargar_features(funcion_extractora):
    positivos = np.load(os.path.join(DATA_DIR, "dataset_real_positivos.npy"))
    negativos = np.load(os.path.join(DATA_DIR, "dataset_real_negativos.npy"))
    X_pos = [funcion_extractora(calcular_espectrograma_stub(s, FS_REAL)) for s in positivos]
    X_neg = [funcion_extractora(calcular_espectrograma_stub(s, FS_REAL)) for s in negativos]
    return np.array(X_pos), np.array(X_neg)

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
    return precision_global, recall_pos, recall_neg

if __name__ == "__main__":
    print("🔬 COMPARACIÓN: Features v1 (3) vs v2 (8) — mismo dataset, leave-one-out")
    print("=" * 78)

    print("\n📦 Cargando y calculando features v1...")
    X_pos_v1, X_neg_v1 = cargar_features(extraer_features_v1)
    print("📦 Cargando y calculando features v2...")
    X_pos_v2, X_neg_v2 = cargar_features(extraer_features_v2)

    configuraciones = [
        (1, False, "K=1, sin balancear"),
        (3, False, "K=3, sin balancear"),
        (1, True,  "K=1, balanceado"),
        (3, True,  "K=3, balanceado"),
        (5, True,  "K=5, balanceado"),
    ]

    print(f"\n{'Config':<25} {'v1 Global':>10} {'v1 R+':>8} {'v1 R-':>8}   {'v2 Global':>10} {'v2 R+':>8} {'v2 R-':>8}")
    print("-" * 90)
    for k, bal, nombre in configuraciones:
        g1, r1p, r1n = leave_one_out(X_pos_v1, X_neg_v1, k, bal)
        g2, r2p, r2n = leave_one_out(X_pos_v2, X_neg_v2, k, bal)
        print(f"{nombre:<25} {g1:>9.1%} {r1p:>7.1%} {r1n:>7.1%}   {g2:>9.1%} {r2p:>7.1%} {r2n:>7.1%}")

    print("\n📋 R+ = Recall positivos (detección de eventos reales, lo que más importa)")
