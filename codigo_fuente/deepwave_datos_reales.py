"""
DeepWave - Validación contra datos reales de LIGO (GW150914)
Descarga, extrae, blanquea (whitening) y prepara el segmento real
del primer evento BBH confirmado de la historia (14 sept 2015).
"""
import numpy as np
import h5py
import urllib.request
import os

URL = "https://gwosc.org/archive/data/O1/1126170624/H-H1_LOSC_4_V1-1126256640-4096.hdf5"
ARCHIVO_LOCAL = "data/GW150914_H1_4096s.hdf5"
GPS_EVENTO = 1126259462.4
GPS_INICIO_ARCHIVO = 1126256640
FS_ORIGINAL = 4096
FS_OBJETIVO = 2048

def descargar_si_falta():
    if not os.path.exists(ARCHIVO_LOCAL):
        os.makedirs("data", exist_ok=True)
        print(f"⬇️  Descargando {URL} ...")
        urllib.request.urlretrieve(URL, ARCHIVO_LOCAL)
        print("✅ Descarga completa.")
    else:
        print("✅ Archivo ya presente localmente.")

def extraer_segmento(duracion_s=1.0):
    with h5py.File(ARCHIVO_LOCAL, "r") as f:
        strain = f["strain"]["Strain"][:]
    offset_s = GPS_EVENTO - GPS_INICIO_ARCHIVO
    centro_muestra = int(offset_s * FS_ORIGINAL)
    mitad_ventana = int((duracion_s / 2) * FS_ORIGINAL)
    segmento = strain[centro_muestra - mitad_ventana : centro_muestra + mitad_ventana]
    print(f"📡 Segmento extraído: {len(segmento)} muestras a {FS_ORIGINAL} Hz")
    print(f"   Amplitud cruda -> min={segmento.min():.3e}, max={segmento.max():.3e}")
    return segmento

def blanquear_y_normalizar(segmento):
    fft = np.fft.rfft(segmento)
    psd_estimado = np.abs(fft) + 1e-30
    fft_blanco = fft / psd_estimado
    señal_blanca = np.fft.irfft(fft_blanco, n=len(segmento))
    señal_norm = señal_blanca / np.std(señal_blanca)
    señal_norm *= 0.3
    return señal_norm

def remuestrear(segmento, fs_orig, fs_obj):
    factor = fs_orig // fs_obj
    return segmento[::factor]

if __name__ == "__main__":
    print("🧠 DEEPWAVE: Validación contra GW150914 (dato REAL, no simulado)")
    print("=" * 60)
    descargar_si_falta()
    segmento_crudo = extraer_segmento(duracion_s=1.0)
    segmento_blanco = blanquear_y_normalizar(segmento_crudo)
    segmento_final = remuestrear(segmento_blanco, FS_ORIGINAL, FS_OBJETIVO)
    print(f"✅ Segmento final: {len(segmento_final)} puntos a {FS_OBJETIVO} Hz")
    print(f"   Amplitud tras whitening -> min={segmento_final.min():.3f}, max={segmento_final.max():.3f}")
    np.save("data/gw150914_procesado.npy", segmento_final)
    print("💾 Guardado en data/gw150914_procesado.npy")
