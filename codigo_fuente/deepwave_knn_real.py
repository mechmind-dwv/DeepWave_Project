"""
DeepWave - Clasificador K-NN nativo (sin TensorFlow) con features
reales extraídas del espectrograma STFT.
Entrenamiento: datos sintéticos etiquetados (único dataset disponible).
Clasificación final: espectrograma de datos REALES de GW150914.
"""
import numpy as np
from deepwave_preprocessing import (
    generar_senal_bbh, generar_senal_glitch, calcular_espectrograma_stub
)

def extraer_features(espectrograma):
    energia_baja = np.mean(espectrograma[:10, :])
    energia_por_tiempo = np.mean(espectrograma, axis=0)
    pendiente = np.polyfit(np.arange(len(energia_por_tiempo)), energia_por_tiempo, 1)[0]
    pico_max = np.max(espectrograma)
    return np.array([energia_baja, pendiente, pico_max])

class DeepWaveKNNReal:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.X_min = None
        self.X_max = None

    def entrenar(self, n_samples=200):
        X, y = [], []
        for _ in range(n_samples // 2):
            señal, fs = generar_senal_bbh()
            spec = calcular_espectrograma_stub(señal, fs)
            X.append(extraer_features(spec))
            y.append(1)
            señal, fs = generar_senal_glitch()
            spec = calcular_espectrograma_stub(señal, fs)
            X.append(extraer_features(spec))
            y.append(0)
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.X_min = self.X_train.min(axis=0)
        self.X_max = self.X_train.max(axis=0)

    def _normalizar(self, X):
        return (X - self.X_min) / (self.X_max - self.X_min + 1e-10)

    def predecir(self, features):
        X_norm = self._normalizar(self.X_train)
        f_norm = self._normalizar(features)
        distancias = np.sum((X_norm - f_norm) ** 2, axis=1)
        k_cercanos = np.argsort(distancias)[:self.k]
        conteo = np.bincount(self.y_train[k_cercanos])
        prediccion = np.argmax(conteo)
        confianza = conteo[prediccion] / self.k
        return prediccion, confianza

if __name__ == "__main__":
    print("🧠 DEEPWAVE K-NN NATIVO: Entrenamiento (sintético) + Clasificación (REAL)")
    print("=" * 70)
    clasificador = DeepWaveKNNReal(k=5)
    print("⚙️  Entrenando con 200 muestras sintéticas etiquetadas...")
    clasificador.entrenar(n_samples=200)
    print("✅ Entrenamiento completo.")
    print("\n📡 Cargando dato REAL: gw150914_whitened_real.npy ...")
    señal_real = np.load("../data/gw150914_whitened_real.npy")
    fs_real = 2048
    spec_real = calcular_espectrograma_stub(señal_real, fs_real)
    features_real = extraer_features(spec_real)
    print(f"   Features extraídas del dato REAL: {features_real}")
    prediccion, confianza = clasificador.predecir(features_real)
    etiqueta = "FUSIÓN BBH (ONDA GRAVITACIONAL) 🌌" if prediccion == 1 else "GLITCH (RUIDO INSTRUMENTAL) 🎧"
    print(f"\n🔬 RESULTADO sobre GW150914 (dato real de LIGO):")
    print(f"   -> {etiqueta}")
    print(f"   -> Confianza (votos K-NN): {confianza:.0%}")
    print("\n📋 NOTA CIENTÍFICA HONESTA:")
    print("   Entrenado exclusivamente con datos sintéticos. El resultado")
    print("   sobre GW150914 es una PRUEBA DE GENERALIZACIÓN, no una")
    print("   validación de detección profesional.")
