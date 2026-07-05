"""
Construye un dataset 100% REAL para entrenar el K-NN:
- Positivos: 11 eventos confirmados de GWTC-1-confident
- Negativos: segmentos de ruido real extraídos de cada archivo,
  lejos en el tiempo del propio evento (dentro del mismo archivo de 32s,
  en los bordes, o de otro evento como ruido de fondo cruzado)

Sin ninguna señal sintética en el entrenamiento.
"""
import numpy as np
import h5py
import os
import warnings
from scipy.signal import welch, butter, filtfilt
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
from gwosc.locate import get_event_urls
from gwosc.datasets import event_gps
import urllib.request

warnings.filterwarnings("ignore")

EVENTOS = ["GW150914-v3", "GW151012-v3", "GW151226-v2", "GW170104-v2",
           "GW170608-v3", "GW170729-v1", "GW170809-v1", "GW170814-v3",
           "GW170817-v3", "GW170818-v1", "GW170823-v1"]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eventos_reales")
FS_OBJETIVO = 2048

def descargar_evento(nombre_evento):
    os.makedirs(DATA_DIR, exist_ok=True)
    urls = get_event_urls(nombre_evento, detector="H1")
    url = urls[0]
    nombre_archivo = os.path.join(DATA_DIR, f"{nombre_evento}.hdf5")
    if not os.path.exists(nombre_archivo):
        print(f"  ⬇️  Descargando {nombre_evento}...")
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

def procesar_evento(nombre_evento):
    """Devuelve (positivo, negativo1, negativo2) - señales reales listas."""
    archivo = descargar_evento(nombre_evento)
    gps_evento = event_gps(nombre_evento)

    with h5py.File(archivo, "r") as f:
        strain = f["strain"]["Strain"][:]
        dt = f["strain"]["Strain"].attrs.get("Xspacing", 1/4096)
        gps_inicio = f["meta"]["GPSstart"][()]

    fs = 1.0 / dt
    señal_blanca = whitening_completo(strain, fs)

    offset_evento = gps_evento - gps_inicio
    positivo = remuestrear(extraer_ventana(señal_blanca, fs, offset_evento), fs, FS_OBJETIVO)

    # Negativos: bordes del mismo archivo de 32s, lejos del evento (>10s de distancia)
    offset_neg1 = max(2, offset_evento - 12)
    offset_neg2 = min(30, offset_evento + 12)
    negativo1 = remuestrear(extraer_ventana(señal_blanca, fs, offset_neg1), fs, FS_OBJETIVO)
    negativo2 = remuestrear(extraer_ventana(señal_blanca, fs, offset_neg2), fs, FS_OBJETIVO)

    return positivo, negativo1, negativo2

if __name__ == "__main__":
    print("🌌 CONSTRUYENDO DATASET 100% REAL (sin síntesis)")
    print("=" * 60)

    X_positivos = []
    X_negativos = []

    for i, evento in enumerate(EVENTOS, 1):
        print(f"\n[{i}/{len(EVENTOS)}] Procesando {evento}...")
        try:
            pos, neg1, neg2 = procesar_evento(evento)
            X_positivos.append(pos)
            X_negativos.append(neg1)
            X_negativos.append(neg2)
            print(f"  ✅ 1 positivo + 2 negativos extraídos")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print(f"\n📊 Dataset final: {len(X_positivos)} positivos reales, {len(X_negativos)} negativos reales")

    np.save(os.path.join(DATA_DIR, "..", "dataset_real_positivos.npy"), np.array(X_positivos))
    np.save(os.path.join(DATA_DIR, "..", "dataset_real_negativos.npy"), np.array(X_negativos))
    print("💾 Guardado en data/dataset_real_positivos.npy y dataset_real_negativos.npy")
