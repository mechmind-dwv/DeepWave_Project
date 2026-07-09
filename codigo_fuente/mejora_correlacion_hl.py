"""
Versión mejorada de la correlación cruzada H1-L1: restringe la
búsqueda del desfase óptimo a la ventana físicamente plausible
(~10ms, el tiempo de viaje de la luz entre Hanford y Livingston),
en vez de buscar en toda la ventana de 1 segundo. Esto debería
reducir las coincidencias espurias de ruido a grandes desfases.
"""
import numpy as np
from scipy.signal import correlate

FS_REAL = 2048
MAX_DESFASE_MS = 10  # tiempo de viaje de la luz H1-L1, ~10ms máximo real
MAX_DESFASE_MUESTRAS = int(MAX_DESFASE_MS / 1000 * FS_REAL)  # ~20 muestras

def correlacion_cruzada_restringida(h1, l1, max_desfase=MAX_DESFASE_MUESTRAS):
    h1_norm = (h1 - h1.mean()) / (h1.std() + 1e-10)
    l1_norm = (l1 - l1.mean()) / (l1.std() + 1e-10)
    n = len(h1_norm)
    corr_completa = correlate(h1_norm, l1_norm, mode="full") / n
    centro = len(corr_completa) // 2
    ventana = corr_completa[centro - max_desfase: centro + max_desfase + 1]
    return np.max(np.abs(ventana))

if __name__ == "__main__":
    import json, os
    DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../data"

    # Recalcular sobre las señales ya guardadas — necesitamos los
    # eventos_reales/*.hdf5 originales, así que reprocesamos H1/L1
    # crudos con el mismo whitening, solo cambia la ventana de búsqueda
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from construir_dataset_l1 import procesar_un_detector

    with open(os.path.join(DATA_DIR, "eventos_procesados.json")) as f:
        registro = json.load(f)
    eventos = [e for e in registro["eventos_procesados"] if e not in
               ["GW190620_030421-v2", "GW190630_185205-v2", "GW190708_232457-v2"]]

    corr_pos_nueva, corr_neg_nueva = [], []
    print(f"🔬 Recalculando correlación H1-L1 con ventana restringida (±{MAX_DESFASE_MS}ms)")
    print("=" * 70)

    for i, evento in enumerate(eventos, 1):
        print(f"[{i}/{len(eventos)}] {evento}...")
        h1 = procesar_un_detector(evento, "H1")
        l1 = procesar_un_detector(evento, "L1")
        c = correlacion_cruzada_restringida(h1, l1)
        corr_pos_nueva.append(c)

        h1n = procesar_un_detector(evento, "H1", offset_extra_s=-12)
        l1n = procesar_un_detector(evento, "L1", offset_extra_s=-12)
        corr_neg_nueva.append(correlacion_cruzada_restringida(h1n, l1n))

        h1n2 = procesar_un_detector(evento, "H1", offset_extra_s=12)
        l1n2 = procesar_un_detector(evento, "L1", offset_extra_s=12)
        corr_neg_nueva.append(correlacion_cruzada_restringida(h1n2, l1n2))

    corr_pos_nueva = np.array(corr_pos_nueva)
    corr_neg_nueva = np.array(corr_neg_nueva)

    print(f"\nCorrelación (ventana restringida) en EVENTOS: media={corr_pos_nueva.mean():.3f}, std={corr_pos_nueva.std():.3f}")
    print(f"Correlación (ventana restringida) en RUIDO: media={corr_neg_nueva.mean():.3f}, std={corr_neg_nueva.std():.3f}")

    from scipy.stats import mannwhitneyu
    stat, p = mannwhitneyu(corr_pos_nueva, corr_neg_nueva, alternative='greater')
    print(f"\nTest Mann-Whitney U: p-valor = {p:.4f}")

    np.save(os.path.join(DATA_DIR, "correlacion_hl_restringida_positivos.npy"), corr_pos_nueva)
    np.save(os.path.join(DATA_DIR, "correlacion_hl_restringida_negativos.npy"), corr_neg_nueva)
    print("\n💾 Guardado en data/correlacion_hl_restringida_*.npy")
