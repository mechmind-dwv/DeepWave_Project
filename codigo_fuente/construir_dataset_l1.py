"""
Amplía el dataset real con el segundo detector (L1), y calcula la
correlación cruzada H1-L1 en la ventana del evento — la pieza que
más se aproxima al método real de confirmación de LIGO/Virgo
(coincidencia entre detectores), ausente en el pipeline solo-H1.
"""
import numpy as np
import h5py
import os
import json
import warnings
from scipy.signal import welch, butter, filtfilt, correlate
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
from gwosc.locate import get_event_urls
from gwosc.datasets import event_gps
import urllib.request

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eventos_reales")
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FS_OBJETIVO = 2048

def descargar_evento(nombre_evento, detector):
    os.makedirs(DATA_DIR, exist_ok=True)
    urls = get_event_urls(nombre_evento, detector=detector)
    url = urls[0]
    nombre_archivo = os.path.join(DATA_DIR, f"{nombre_evento}_{detector}.hdf5")
    if not os.path.exists(nombre_archivo):
        print(f"  ⬇️  Descargando {nombre_evento} ({detector})...")
        urllib.request.urlretrieve(url, nombre_archivo)
    return nombre_archivo

def whitening_completo(segmento, fs):
    nperseg = min(int(fs * 4), len(segmento))
    freqs, psd = welch(segmento, fs=fs, nperseg=nperseg, window="hann")
    interp_psd = interp1d(freqs, psd, bounds_error=False, fill_value="extrapolate")
    n = len(segmento)
    ventana = tukey(n, alpha=0.2)
    hf = np.fft.rfft(segmento * ventana)
    freqs_full = np.fft.rfftfreq(n, 1/fs)
    norm = 1.0 / np.sqrt(fs / 2)
    hf_blanco = hf / np.sqrt(interp_psd(freqs_full)) * norm
    blanco = np.fft.irfft(hf_blanco, n=n)
    nyq = fs / 2
    b, a = butter(4, [35 / nyq, 350 / nyq], btype="band")
    return filtfilt(b, a, blanco)

def extraer_ventana(señal_blanca, fs, centro_s, duracion_s=1):
    centro_muestra = int(centro_s * fs)
    mitad = int((duracion_s / 2) * fs)
    return señal_blanca[centro_muestra - mitad: centro_muestra + mitad]

def remuestrear(segmento, fs_orig, fs_obj):
    factor = int(fs_orig) // fs_obj
    return segmento[::factor]

def procesar_un_detector(nombre_evento, detector, offset_extra_s=0):
    """offset_extra_s permite extraer negativos igual que antes."""
    archivo = descargar_evento(nombre_evento, detector)
    gps_evento = event_gps(nombre_evento)
    with h5py.File(archivo, "r") as f:
        strain = f["strain"]["Strain"][:]
        dt = f["strain"]["Strain"].attrs.get("Xspacing", 1/4096)
        gps_inicio = f["meta"]["GPSstart"][()]
    if np.isnan(strain).any():
        return None  # dato incompleto, igual que GW190425
    fs = 1.0 / dt
    señal_blanca = whitening_completo(strain, fs)
    offset = (gps_evento - gps_inicio) + offset_extra_s
    offset = max(2, min(30, offset))
    ventana = extraer_ventana(señal_blanca, fs, offset)
    return remuestrear(ventana, fs, FS_OBJETIVO)

def correlacion_cruzada_normalizada(h1, l1):
    """Máximo de correlación cruzada normalizada entre H1 y L1 — una
    señal real coincidente en ambos detectores debería tener un pico
    de correlación notablemente mayor que ruido no correlacionado."""
    h1_norm = (h1 - h1.mean()) / (h1.std() + 1e-10)
    l1_norm = (l1 - l1.mean()) / (l1.std() + 1e-10)
    corr = correlate(h1_norm, l1_norm, mode="full") / len(h1_norm)
    return np.max(np.abs(corr))

def procesar_evento_dual(nombre_evento):
    h1_pos = procesar_un_detector(nombre_evento, "H1")
    l1_pos = procesar_un_detector(nombre_evento, "L1")
    if h1_pos is None or l1_pos is None:
        raise ValueError("Dato incompleto (NaN) en H1 o L1")

    h1_neg1 = procesar_un_detector(nombre_evento, "H1", offset_extra_s=-12)
    l1_neg1 = procesar_un_detector(nombre_evento, "L1", offset_extra_s=-12)
    h1_neg2 = procesar_un_detector(nombre_evento, "H1", offset_extra_s=12)
    l1_neg2 = procesar_un_detector(nombre_evento, "L1", offset_extra_s=12)

    return {
        "positivo": (h1_pos, l1_pos, correlacion_cruzada_normalizada(h1_pos, l1_pos)),
        "negativo1": (h1_neg1, l1_neg1, correlacion_cruzada_normalizada(h1_neg1, l1_neg1)),
        "negativo2": (h1_neg2, l1_neg2, correlacion_cruzada_normalizada(h1_neg2, l1_neg2)),
    }

if __name__ == "__main__":
    with open(os.path.join(DATASET_DIR, "eventos_procesados.json")) as f:
        registro = json.load(f)
    eventos = registro["eventos_procesados"]

    resultados_corr_positivo = []
    resultados_corr_negativo = []
    exitosos, fallidos = 0, 0

    for i, evento in enumerate(eventos, 1):
        print(f"\n[{i}/{len(eventos)}] Procesando {evento} (H1+L1)...")
        try:
            r = procesar_evento_dual(evento)
            resultados_corr_positivo.append(r["positivo"][2])
            resultados_corr_negativo.append(r["negativo1"][2])
            resultados_corr_negativo.append(r["negativo2"][2])
            exitosos += 1
            print(f"  ✅ Correlación H1-L1 (evento): {r['positivo'][2]:.3f}")
        except Exception as e:
            fallidos += 1
            print(f"  ❌ Error: {e}")

    print(f"\n📊 Procesados: {exitosos}, fallidos: {fallidos}")
    print(f"\nCorrelación H1-L1 en EVENTOS reales: media={np.mean(resultados_corr_positivo):.3f}, std={np.std(resultados_corr_positivo):.3f}")
    print(f"Correlación H1-L1 en RUIDO: media={np.mean(resultados_corr_negativo):.3f}, std={np.std(resultados_corr_negativo):.3f}")

    np.save(os.path.join(DATASET_DIR, "correlacion_hl_positivos.npy"), np.array(resultados_corr_positivo))
    np.save(os.path.join(DATASET_DIR, "correlacion_hl_negativos.npy"), np.array(resultados_corr_negativo))
    print("\n💾 Correlaciones guardadas en data/correlacion_hl_*.npy")
