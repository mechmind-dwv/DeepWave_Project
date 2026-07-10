"""
Compara el SNR (relación señal-ruido) de los eventos originales
(GWTC-1/2.1/3, primeros 75) contra los añadidos recientemente
(GWTC-4/5, los últimos 32), para verificar si el descenso del AUC se
debe a que los nuevos eventos son sistemáticamente más difíciles.
"""
import json
import warnings
from gwosc.api import fetch_event_json
from scipy.stats import mannwhitneyu
import numpy as np

warnings.filterwarnings("ignore")

with open("data/eventos_procesados.json") as f:
    registro = json.load(f)

eventos = registro["eventos_procesados"]
originales = eventos[:75]
nuevos = eventos[75:]

print(f"Eventos originales (GWTC-1/2.1/3): {len(originales)}")
print(f"Eventos nuevos (GWTC-4/5): {len(nuevos)}\n")


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


print("Consultando SNR de eventos originales...")
snr_originales = []
for e in originales:
    snr = obtener_snr(e)
    if snr is not None:
        snr_originales.append(snr)

print("Consultando SNR de eventos nuevos...")
snr_nuevos = []
for e in nuevos:
    snr = obtener_snr(e)
    if snr is not None:
        snr_nuevos.append(snr)

snr_originales = np.array(snr_originales)
snr_nuevos = np.array(snr_nuevos)

print(f"\n📊 SNR originales: n={len(snr_originales)}, media={snr_originales.mean():.2f}, mediana={np.median(snr_originales):.2f}")
print(f"📊 SNR nuevos:      n={len(snr_nuevos)}, media={snr_nuevos.mean():.2f}, mediana={np.median(snr_nuevos):.2f}")

if len(snr_originales) > 0 and len(snr_nuevos) > 0:
    stat, p = mannwhitneyu(snr_originales, snr_nuevos, alternative="greater")
    print(f"\nTest Mann-Whitney U (H1: originales > nuevos): p-valor = {p:.4f}")
    if p < 0.05:
        print("✅ Los eventos nuevos SÍ tienen SNR significativamente menor")
    else:
        print("⚠️  No hay diferencia significativa de SNR entre tandas")

np.save("data/snr_originales.npy", snr_originales)
np.save("data/snr_nuevos.npy", snr_nuevos)
