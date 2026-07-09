"""
v6: combina las 2 features de mayor correlación individual del
análisis de importancia (pico_max, energia_media) con la correlación
H1-L1 de ventana amplia (v4, la de mejor desempeño práctico hasta
ahora). 3 features cuidadosamente seleccionadas, no acumuladas.
"""
import numpy as np
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codigo_fuente.deepwave_preprocessing import calcular_espectrograma_stub
from codigo_fuente.deepwave_features_v2 import extraer_features_v2
from codigo_fuente.experimentar_knn_real import leave_one_out

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FS_REAL = 2048

def extraer_pico_y_energia_media(espectrograma):
    """Solo las 2 features con mayor correlación individual:
    pico_max (+0.569) y energia_media (+0.348)."""
    full = extraer_features_v2(espectrograma)
    # v2 devuelve: [energia_baja, energia_media, energia_alta, pendiente,
    #               pico_max, varianza_total, entropia_espectral, num_picos]
    return np.array([full[1], full[4]])  # energia_media, pico_max

if __name__ == "__main__":
    with open(os.path.join(DATA_DIR, "eventos_procesados.json")) as f:
        registro = json.load(f)
    eventos_todos = registro["eventos_procesados"]
    excluidos = ["GW190620_030421-v2", "GW190630_185205-v2", "GW190708_232457-v2"]
    eventos_con_hl = [e for e in eventos_todos if e not in excluidos]

    positivos = np.load(os.path.join(DATA_DIR, "dataset_real_positivos.npy"))
    negativos = np.load(os.path.join(DATA_DIR, "dataset_real_negativos.npy"))
    corr_pos = np.load(os.path.join(DATA_DIR, "correlacion_hl_positivos.npy"))
    corr_neg = np.load(os.path.join(DATA_DIR, "correlacion_hl_negativos.npy"))

    indices_con_hl = [i for i, e in enumerate(eventos_todos) if e in eventos_con_hl]
    positivos_sub = positivos[indices_con_hl]
    indices_neg = []
    for i in indices_con_hl:
        indices_neg.extend([2*i, 2*i+1])
    negativos_sub = negativos[indices_neg]

    X_pos_selectas = np.array([extraer_pico_y_energia_media(calcular_espectrograma_stub(s, FS_REAL)) for s in positivos_sub])
    X_neg_selectas = np.array([extraer_pico_y_energia_media(calcular_espectrograma_stub(s, FS_REAL)) for s in negativos_sub])

    X_pos_v6 = np.hstack([X_pos_selectas, corr_pos.reshape(-1, 1)])
    X_neg_v6 = np.hstack([X_neg_selectas, corr_neg.reshape(-1, 1)])

    print("🔬 v6: pico_max + energia_media + correlación H1-L1 (3 features selectas)")
    print("=" * 75)
    configuraciones = [(1, False), (3, False), (1, True), (3, True), (5, True)]
    print(f"{'Config':<20} {'v6 Global':>10} {'v6 Recall+':>12} {'v6 Recall-':>12}")
    print("-" * 60)
    mejor_r6 = 0
    for k, bal in configuraciones:
        g6, r6p, r6n, _, _ = leave_one_out(X_pos_v6, X_neg_v6, k, bal)
        nombre = f"K={k}, {'bal' if bal else 'sin bal'}"
        print(f"{nombre:<20} {g6:>9.1%} {r6p:>11.1%} {r6n:>11.1%}")
        mejor_r6 = max(mejor_r6, r6p)

    print(f"\n📋 Comparación final:")
    print(f"   v1 (solo H1, 3 feat.):        Recall+ máx = 67.5% (dataset 40 eventos)")
    print(f"   v4 (H1 + corr. HL amplia):     Recall+ máx = 67.6% (subconjunto 37 eventos)")
    print(f"   v6 (2 feat. selectas + corr.): Recall+ máx = {mejor_r6:.1%} (subconjunto 37 eventos)")
