"""
Valida el modelo MLP entrenado contra datos REALES de LIGO (GW150914)
y controles negativos de ruido real, mismo estándar que el K-NN.
"""
import numpy as np
from joblib import load
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepwave_preprocessing import calcular_espectrograma_stub
from deepwave_control_negativo import procesar_segmento

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modelo_mlp.joblib")
FS_REAL = 2048
GPS_EVENTO = 1126259462.4
GPS_INICIO_ARCHIVO = 1126256640

def clasificar_con_mlp(senal, bundle):
    spec = calcular_espectrograma_stub(senal, FS_REAL)
    X = spec.flatten().reshape(1, -1)
    X_scaled = bundle['scaler'].transform(X)
    prob = bundle['model'].predict_proba(X_scaled)[0][1]
    is_bbh = prob > 0.5
    confianza = prob if is_bbh else 1.0 - prob
    return is_bbh, confianza

if __name__ == "__main__":
    print("🔬 VALIDACIÓN MLP: GW150914 real + controles negativos reales")
    print("=" * 65)

    bundle = load(MODEL_PATH)
    ruta_datos = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

    señal_evento = np.load(os.path.join(ruta_datos, "gw150914_whitened_real.npy"))
    is_bbh, conf = clasificar_con_mlp(señal_evento, bundle)
    etiqueta = "FUSIÓN BBH 🌌" if is_bbh else "GLITCH/RUIDO 🎧"
    print(f"\n[GW150914 - evento real] -> {etiqueta} (confianza {conf:.1%})")

    offset_evento = GPS_EVENTO - GPS_INICIO_ARCHIVO
    offsets = [
        ("RUIDO -1000s", offset_evento - 1000),
        ("RUIDO +500s", offset_evento + 500),
    ]
    print("\n🔬 Controles negativos (ruido real, lejos del evento):")
    for nombre, offset in offsets:
        segmento = procesar_segmento(offset)
        is_bbh_n, conf_n = clasificar_con_mlp(segmento, bundle)
        etiqueta_n = "FUSIÓN BBH 🌌" if is_bbh_n else "GLITCH/RUIDO 🎧"
        print(f"[{nombre}] -> {etiqueta_n} (confianza {conf_n:.1%})")
