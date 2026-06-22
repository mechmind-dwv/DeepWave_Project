import numpy as np

class DeepWaveClassifier:
    def __init__(self):
        self.X_train = np.array([
            [0.1, 50, 0.9], [0.8, 150, 0.2], [0.2, 55, 0.85],
            [0.9, 145, 0.25], [0.05, 40, 0.95], [1.0, 160, 0.15]
        ])
        self.y_train = np.array([1, 0, 1, 0, 1, 0])
        self.K = 3

    def _calcular_distancias(self, evento):
        X_min = self.X_train.min(axis=0)
        X_max = self.X_train.max(axis=0)
        X_norm = (self.X_train - X_min) / (X_max - X_min)
        evento_norm = (np.array(evento) - X_min) / (X_max - X_min)
        distancias_sq = np.sum((X_norm - evento_norm)**2, axis=1)
        return distancias_sq

    def predecir_evento(self, features):
        distancias_sq = self._calcular_distancias(features)
        indices_k_cercanos = np.argsort(distancias_sq)[:self.K]
        etiquetas_k_cercanas = self.y_train[indices_k_cercanos]
        conteo = np.bincount(etiquetas_k_cercanas)
        prediccion = np.argmax(conteo)
        return "FUSIÓN BBH (ONDA GRAVITACIONAL) 🌌" if prediccion == 1 else "GLITCH (RUIDO INSTRUMENTAL) 🎧"

if __name__ == "__main__":
    detector = DeepWaveClassifier()
    print("🌟 Iniciando DeepWave - Clasificador de Ondas Gravitacionales")
    print("-------------------------------------------------------")
    print(f"Modelo K-NN ({detector.K} vecinos) listo para clasificar.")
    evento_real = [0.15, 60, 0.9]
    print(f"\n🔬 Análisis [Amplitud:0.15, Frecuencia:60Hz, Persistencia:0.9]:")
    print(f"  -> Resultado: {detector.predecir_evento(evento_real)}")
    evento_ruido = [0.95, 140, 0.1]
    print(f"\n🔬 Análisis [Amplitud:0.95, Frecuencia:140Hz, Persistencia:0.1]:")
    print(f"  -> Resultado: {detector.predecir_evento(evento_ruido)}")
    evento_intermedio = [0.5, 100, 0.5]
    print(f"\n🔬 Análisis [Amplitud:0.5, Frecuencia:100Hz, Persistencia:0.5]:")
    print(f"  -> Resultado: {detector.predecir_evento(evento_intermedio)}")
