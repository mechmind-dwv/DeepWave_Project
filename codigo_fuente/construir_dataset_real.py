"""
Construye/amplía un dataset 100% REAL para entrenar el K-NN.
Acepta una lista de eventos, evita reprocesar los ya guardados,
y AÑADE al dataset existente en vez de sobrescribirlo.
"""
import numpy as np
import h5py
import os
import json
import warnings
from scipy.signal import welch, butter, filtfilt
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
from gwosc.locate import get_event_urls
from gwosc.datasets import event_gps
import urllib.request

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eventos_reales")
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
REGISTRO_PATH = os.path.join(DATASET_DIR, "eventos_procesados.json")
FS_OBJETIVO = 2048

def cargar_registro():
    """Devuelve (lista_procesados, lista_excluidos). Compatible con
    el formato antiguo (lista simple) y el nuevo (dict con dos claves)."""
    if os.path.exists(REGISTRO_PATH):
        with open(REGISTRO_PATH, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get("eventos_procesados", []), data.get("eventos_excluidos_por_nan", [])
        else:
            return data, []
    return [], []

def guardar_registro(procesados, excluidos):
    with open(REGISTRO_PATH, "w") as f:
        json.dump({"eventos_procesados": procesados, "eventos_excluidos_por_nan": excluidos}, f, indent=2)

def descargar_evento(nombre_evento):
    os.makedirs(DATA_DIR, exist_ok=True)
    urls = get_event_urls(nombre_evento, detector="H1")
    if not urls:
        urls = get_event_urls(nombre_evento, detector="L1")
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
    offset_neg1 = max(2, offset_evento - 12)
    offset_neg2 = min(30, offset_evento + 12)
    negativo1 = remuestrear(extraer_ventana(señal_blanca, fs, offset_neg1), fs, FS_OBJETIVO)
    negativo2 = remuestrear(extraer_ventana(señal_blanca, fs, offset_neg2), fs, FS_OBJETIVO)
    return positivo, negativo1, negativo2

def ampliar_dataset(nuevos_eventos):
    procesados, excluidos = cargar_registro()

    ruta_pos = os.path.join(DATASET_DIR, "dataset_real_positivos.npy")
    ruta_neg = os.path.join(DATASET_DIR, "dataset_real_negativos.npy")
    positivos = list(np.load(ruta_pos)) if os.path.exists(ruta_pos) else []
    negativos = list(np.load(ruta_neg)) if os.path.exists(ruta_neg) else []

    eventos_a_procesar = [e for e in nuevos_eventos if e not in procesados and e not in excluidos]
    print(f"📋 {len(eventos_a_procesar)} eventos nuevos de {len(nuevos_eventos)} solicitados (resto ya procesado)")

    exitosos, fallidos = 0, 0
    for i, evento in enumerate(eventos_a_procesar, 1):
        print(f"\n[{i}/{len(eventos_a_procesar)}] Procesando {evento}...")
        try:
            pos, neg1, neg2 = procesar_evento(evento)
            if np.isnan(pos).any() or np.isnan(neg1).any() or np.isnan(neg2).any():
                excluidos.append(evento)
                fallidos += 1
                print(f"  ⚠️  Excluido: contiene NaN (dato real incompleto)")
                continue
            positivos.append(pos)
            negativos.append(neg1)
            negativos.append(neg2)
            procesados.append(evento)
            exitosos += 1
            print(f"  ✅ 1 positivo + 2 negativos extraídos")
        except Exception as e:
            fallidos += 1
            print(f"  ❌ Error: {e}")

        # Guardado incremental cada 5 eventos: si la sesión se corta,
        # como máximo se pierden 4 eventos ya procesados, no todos.
        if i % 5 == 0:
            np.save(ruta_pos, np.array(positivos))
            np.save(ruta_neg, np.array(negativos))
            guardar_registro(procesados, excluidos)
            print(f"  💾 Guardado incremental ({len(positivos)} positivos hasta ahora)")

    np.save(ruta_pos, np.array(positivos))
    np.save(ruta_neg, np.array(negativos))
    guardar_registro(procesados, excluidos)

    print(f"\n📊 Dataset TOTAL ahora: {len(positivos)} positivos, {len(negativos)} negativos")
    print(f"   ({exitosos} exitosos, {fallidos} fallidos en esta ronda)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python construir_dataset_real.py evento1 evento2 ...")
        sys.exit(1)
    ampliar_dataset(sys.argv[1:])
