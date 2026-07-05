"""
Control negativo: extrae un segmento de RUIDO REAL (sin evento conocido),
lejos en el tiempo de GW150914, y lo pasa por el mismo pipeline de
whitening + K-NN. Si el clasificador también dice "BBH" aquí, el
resultado anterior era un falso positivo del modelo, no una detección real.
"""
import numpy as np
import h5py
from scipy.signal import welch, butter, filtfilt
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d
import sys
sys.path.insert(0, "codigo_fuente")
from deepwave_preprocessing import calcular_espectrograma_stub
from deepwave_knn_real import DeepWaveKNNReal, extraer_features

ARCHIVO_LOCAL = "data/GW150914_H1_4096s.hdf5"
GPS_EVENTO = 1126259462.4
GPS_INICIO_ARCHIVO = 1126256640
FS_ORIGINAL = 4096
FS_OBJETIVO = 2048

def procesar_segmento(centro_offset_s, duracion_analisis_s=32, duracion_final_s=1):
    with h5py.File(ARCHIVO_LOCAL, "r") as f:
        strain = f["strain"]["Strain"][:]
        dt = f["strain"]["Strain"].attrs.get("Xspacing", 1.0 / FS_ORIGINAL)
    fs = 1.0 / dt
    centro = int(centro_offset_s * FS_ORIGINAL)
    mitad = int((duracion_analisis_s / 2) * FS_ORIGINAL)
    segmento = strain[centro - mitad: centro + mitad]

    nperseg = int(fs * 4)
    freqs, psd = welch(segmento, fs=fs, nperseg=nperseg, window="hann")
    interp_psd = interp1d(freqs, psd, bounds_error=False, fill_value="extrapolate")

    n = len(segmento)
    ventana = tukey(n, alpha=0.2)
    hf = np.fft.rfft(segmento * ventana)
    freqs_full = np.fft.rfftfreq(n, dt)
    norm = 1.0 / np.sqrt(1.0 / (dt * 2))
    hf_blanco = hf / np.sqrt(interp_psd(freqs_full)) * norm
    blanco = np.fft.irfft(hf_blanco, n=n)

    nyq = fs / 2
    b, a = butter(4, [35 / nyq, 350 / nyq], btype="band")
    filtrado = filtfilt(b, a, blanco)

    centro_v = len(filtrado) // 2
    mitad_v = int((duracion_final_s / 2) * fs)
    ventana_final = filtrado[centro_v - mitad_v: centro_v + mitad_v]
    factor = int(fs) // FS_OBJETIVO
    return ventana_final[::factor]

if __name__ == "__main__":
    print("🔬 CONTROL NEGATIVO: ruido real, lejos del evento GW150914")
    print("=" * 65)

    offset_evento = GPS_EVENTO - GPS_INICIO_ARCHIVO
    offset_ruido_1 = offset_evento - 1000   # 1000s antes del evento
    offset_ruido_2 = offset_evento + 500   # 1500s después

    clasificador = DeepWaveKNNReal(k=5)
    clasificador.entrenar(n_samples=200)

    for nombre, offset in [("RUIDO -1000s", offset_ruido_1), ("RUIDO +500s", offset_ruido_2)]:
        segmento = procesar_segmento(offset)
        spec = calcular_espectrograma_stub(segmento, FS_OBJETIVO)
        features = extraer_features(spec)
        pred, conf = clasificador.predecir(features)
        etiqueta = "FUSIÓN BBH 🌌" if pred == 1 else "GLITCH/RUIDO 🎧"
        print(f"\n[{nombre}] features={features}")
        print(f"   -> {etiqueta} (confianza {conf:.0%})")

    print("\n📋 INTERPRETACIÓN:")
    print("   Si estos segmentos de ruido normal también salen 'BBH',")
    print("   el resultado sobre GW150914 NO es una detección real —")
    print("   es un sesgo del clasificador hacia esa clase.")
