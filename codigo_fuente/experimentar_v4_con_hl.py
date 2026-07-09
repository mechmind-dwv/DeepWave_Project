"""
K-NN con 4 features: las 3 originales de v1 + correlación H1-L1.
Solo sobre los 37 eventos que tienen ambos detectores disponibles
(excluye los 3 sin H1).
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
    with open(os.path.join(DATA_DIR, "eventos_procesados.json")) as f:
        registro = json.load(f)
    eventos_todos = registro["eventos_procesados"]
    eventos_con_hl = [e for e in eventos_todos if e not in
                      ["GW190620_030421-v2", "GW190630_185205-v2", "GW190708_232457-v2"]]

    positivos = np.load(os.path.join(DATA_DIR, "dataset_real_positivos.npy"))
    negativos = np.load(os.path.join(DATA_DIR, "dataset_real_negativos.npy"))
    corr_pos = np.load(os.path.join(DATA_DIR, "correlacion_hl_positivos.npy"))
    corr_neg = np.load(os.path.join(DATA_DIR, "correlacion_hl_negativos.npy"))

    # Las correlaciones se guardaron en el mismo orden que eventos_con_hl (37 eventos)
    # positivos/negativos tienen 40/80 — hay que tomar el subconjunto correspondiente
    indices_con_hl = [i for i, e in enumerate(eventos_todos) if e in eventos_con_hl]
    positivos_sub = positivos[indices_con_hl]
    # negativos: 2 por evento, mismo orden
    indices_neg = []
    for i in indices_con_hl:
        indices_neg.extend([2*i, 2*i+1])
    negativos_sub = negativos[indices_neg]

    print(f"Subconjunto: {len(positivos_sub)} positivos, {len(negativos_sub)} negativos")

    # Features v1 (3) + correlación H1-L1 (1) = 4 features
    X_pos_v1 = np.array([extraer_features(calcular_espectrograma_stub(s, FS_REAL)) for s in positivos_sub])
    X_neg_v1 = np.array([extraer_features(calcular_espectrograma_stub(s, FS_REAL)) for s in negativos_sub])

    X_pos_v4 = np.hstack([X_pos_v1, corr_pos.reshape(-1, 1)])
    X_neg_v4 = np.hstack([X_neg_v1, corr_neg.reshape(-1, 1)])

    print("\n🔬 COMPARACIÓN: v1 (solo H1) vs v4 (H1 + correlación H1-L1), mismo subconjunto")
    print("=" * 80)

    configuraciones = [(1, False), (3, False), (1, True), (3, True), (5, True)]
    print(f"{'Config':<20} {'v1 Global':>10} {'v1 R+':>8}   {'v4 Global':>10} {'v4 R+':>8}")
    print("-" * 65)
    for k, bal in configuraciones:
        g1, r1p, _, _, _ = leave_one_out(X_pos_v1, X_neg_v1, k, bal)
        g4, r4p, _, _, _ = leave_one_out(X_pos_v4, X_neg_v4, k, bal)
        nombre = f"K={k}, {'bal' if bal else 'sin bal'}"
        print(f"{nombre:<20} {g1:>9.1%} {r1p:>7.1%}   {g4:>9.1%} {r4p:>7.1%}")
