"""
Calcula el AUC del clasificador v1 separando el dataset en dos grupos
por SNR (alto vs bajo), para verificar si el rendimiento es realmente
estable DENTRO de cada rango de dificultad — la hipótesis que
explicaría por qué el AUC agregado bajó al mezclar eventos O4 (SNR
más bajo) con los originales.
"""
import numpy as np
import json
import os
import sys
import warnings
from gwosc.api import fetch_event_json
from scipy.stats import rankdata

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codigo_fuente.deepwave_preprocessing import calcular_espectrograma_stub
from codigo_fuente.deepwave_knn_real import extraer_features

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FS_REAL = 2048
K_SCORE = 15


def obtener_snr(nombre_evento):
    try:
        data = fetch_event_json(nombre_evento)
        for version, info in data.get("events", {}).items():
            snr = info.get("network_matched_filter_snr")
            if snr is not None:
                return float(snr)
    except Exception:
        return None
    return None


def score_continuo(X_train, y_train, features_test, k):
    X_min, X_max = X_train.min(axis=0), X_train.max(axis=0)
    X_norm = (X_train - X_min) / (X_max - X_min + 1e-10)
    f_norm = (features_test - X_min) / (X_max - X_min + 1e-10)
    distancias = np.sum((X_norm - f_norm) ** 2, axis=1)
    k_cercanos = np.argsort(distancias)[:k]
    return np.mean(y_train[k_cercanos] == 1)


def auc_mann_whitney(scores, etiquetas):
    rangos = rankdata(scores)
    n_pos = np.sum(etiquetas == 1)
    n_neg = np.sum(etiquetas == 0)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    suma_rangos_pos = np.sum(rangos[etiquetas == 1])
    u = suma_rangos_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


if __name__ == "__main__":
    with open(os.path.join(DATA_DIR, "eventos_procesados.json")) as f:
        registro = json.load(f)
    eventos = registro["eventos_procesados"]

    print("Consultando SNR de todos los eventos...")
    snrs = {}
    for e in eventos:
        s = obtener_snr(e)
        if s is not None:
            snrs[e] = s

    mediana_snr = np.median(list(snrs.values()))
    print(f"Mediana SNR global: {mediana_snr:.2f}\n")

    eventos_alto_snr = [e for e in eventos if snrs.get(e, 0) >= mediana_snr]
    eventos_bajo_snr = [e for e in eventos if e in snrs and snrs.get(e, 0) < mediana_snr]

    print(f"Grupo SNR alto (>={mediana_snr:.2f}): {len(eventos_alto_snr)} eventos")
    print(f"Grupo SNR bajo (<{mediana_snr:.2f}):  {len(eventos_bajo_snr)} eventos\n")

    positivos = np.load(os.path.join(DATA_DIR, "dataset_real_positivos.npy"))
    negativos = np.load(os.path.join(DATA_DIR, "dataset_real_negativos.npy"))

    def calcular_auc_grupo(indices_eventos):
        X_pos_grupo = np.array([extraer_features(calcular_espectrograma_stub(positivos[i], FS_REAL)) for i in indices_eventos])
        # negativos correspondientes (2 por evento)
        indices_neg = []
        for i in indices_eventos:
            indices_neg.extend([2 * i, 2 * i + 1])
        X_neg_grupo = np.array([extraer_features(calcular_espectrograma_stub(negativos[i], FS_REAL)) for i in indices_neg])

        X_todo = np.vstack([X_pos_grupo, X_neg_grupo])
        y_todo = np.array([1] * len(X_pos_grupo) + [0] * len(X_neg_grupo))

        scores = []
        for i in range(len(X_todo)):
            X_train = np.delete(X_todo, i, axis=0)
            y_train = np.delete(y_todo, i)
            scores.append(score_continuo(X_train, y_train, X_todo[i], min(K_SCORE, len(X_train) - 1)))

        return auc_mann_whitney(np.array(scores), y_todo), len(X_pos_grupo)

    indices_alto = [i for i, e in enumerate(eventos) if e in eventos_alto_snr]
    indices_bajo = [i for i, e in enumerate(eventos) if e in eventos_bajo_snr]

    auc_alto, n_alto = calcular_auc_grupo(indices_alto)
    auc_bajo, n_bajo = calcular_auc_grupo(indices_bajo)

    print(f"📊 AUC en grupo SNR ALTO (n={n_alto}): {auc_alto:.3f}")
    print(f"📊 AUC en grupo SNR BAJO (n={n_bajo}): {auc_bajo:.3f}")
    print(f"\n📋 AUC agregado (sin condicionar, referencia): 0.687 (n=107)")
