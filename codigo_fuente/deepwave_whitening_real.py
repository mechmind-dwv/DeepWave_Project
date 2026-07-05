"""
DeepWave - Whitening real siguiendo el método oficial documentado
por GWOSC/LOSC para datos públicos: PSD por Welch + normalización
espectral + filtro pasa-banda Butterworth.
"""
import numpy as np
import h5py
from scipy.signal import welch, butter, filtfilt
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d

ARCHIVO_LOCAL = "data/GW150914_H1_4096s.hdf5"
GPS_EVENTO = 1126259462.4
GPS_INICIO_ARCHIVO = 1126256640
FS_ORIGINAL = 4096
FS_OBJETIVO = 2048
DURACION_ANALISIS_S = 32
DURACION_EXTRACCION_S = 1

def cargar_segmento_amplio(duracion_s=DURACION_ANALISIS_S):
    with h5py.File(ARCHIVO_LOCAL, "r") as f:
        strain = f["strain"]["Strain"][:]
        dt = f["strain"]["Strain"].attrs.get("Xspacing", 1.0 / FS_ORIGINAL)
    offset_s = GPS_EVENTO - GPS_INICIO_ARCHIVO
    centro = int(offset_s * FS_ORIGINAL)
    mitad = int((duracion_s / 2) * FS_ORIGINAL)
    segmento = strain[centro - mitad: centro + mitad]
    return segmento, dt

def estimar_psd_welch(segmento, fs):
    nperseg = int(fs * 4)
    freqs, psd = welch(segmento, fs=fs, nperseg=nperseg, window="hann")
    return interp1d(freqs, psd, bounds_error=False, fill_value="extrapolate")

def whiten(segmento, interp_psd, dt):
    n = len(segmento)
    ventana = tukey(n, alpha=0.2)
    segmento_ventaneado = segmento * ventana
    freqs = np.fft.rfftfreq(n, dt)
    hf = np.fft.rfft(segmento_ventaneado)
    norm = 1.0 / np.sqrt(1.0 / (dt * 2))
    hf_blanco = hf / np.sqrt(interp_psd(freqs)) * norm
    return np.fft.irfft(hf_blanco, n=n)

def filtro_bandpass(segmento, fs, f_bajo=35, f_alto=350, orden=4):
    nyq = fs / 2
    b, a = butter(orden, [f_bajo / nyq, f_alto / nyq], btype="band")
    return filtfilt(b, a, segmento)

def extraer_ventana_final(segmento_procesado, fs, duracion_s=DURACION_EXTRACCION_S):
    centro = len(segmento_procesado) // 2
    mitad = int((duracion_s / 2) * fs)
    return segmento_procesado[centro - mitad: centro + mitad]

def remuestrear(segmento, fs_orig, fs_obj):
    factor = fs_orig // fs_obj
    return segmento[::factor]

if __name__ == "__main__":
    print("🔬 DEEPWAVE: Whitening REAL (método oficial GWOSC/LOSC)")
    print("=" * 60)
    segmento_crudo, dt = cargar_segmento_amplio()
    fs = 1.0 / dt
    print(f"📡 Segmento cargado: {len(segmento_crudo)} muestras, fs={fs:.1f} Hz")
    print(f"   Amplitud cruda -> min={segmento_crudo.min():.3e}, max={segmento_crudo.max():.3e}")
    print("\n⚙️  Estimando PSD real por método de Welch (nperseg=4s)...")
    interp_psd = estimar_psd_welch(segmento_crudo, fs)
    print("⚙️  Aplicando whitening espectral...")
    segmento_blanco = whiten(segmento_crudo, interp_psd, dt)
    print("⚙️  Aplicando filtro pasa-banda 35-350 Hz...")
    segmento_filtrado = filtro_bandpass(segmento_blanco, fs)
    print("⚙️  Extrayendo ventana de 1s centrada en el evento...")
    ventana_final = extraer_ventana_final(segmento_filtrado, fs)
    print("⚙️  Remuestreando a 2048 Hz...")
    segmento_final = remuestrear(ventana_final, int(fs), FS_OBJETIVO)
    print(f"\n✅ Segmento final: {len(segmento_final)} puntos a {FS_OBJETIVO} Hz")
    print(f"   Amplitud tras whitening real -> min={segmento_final.min():.3f}, max={segmento_final.max():.3f}, std={segmento_final.std():.3f}")
    np.save("data/gw150914_whitened_real.npy", segmento_final)
    print("💾 Guardado en data/gw150914_whitened_real.npy")
