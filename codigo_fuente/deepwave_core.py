"""
MÓDULO PRINCIPAL DEEPWAVE - Detector de Ondas Gravitacionales
Autores: Benjamin Cabeza Duran / Gemini IA
Fecha: Octubre 2025
"""

import numpy as np

class DeepWaveClassifier:
    def __init__(self):
        # Datos simulados: Features son 
        # [Amplitud RMS, Frecuencia Pico, Persistencia]
        # Etiquetas: 1=BBH Fusión, 0=Glitch (Ruido)
        self.X_train = np.array([
            [0.1, 50, 0.9], [0.8, 150, 0.2], [0.2, 55, 0.85], 
            [0.9, 145, 0.25], [0.05, 40, 0.95], [1.0, 160, 0.15]
        ])
        self.y_train = np.array([1, 0, 1, 0, 1, 0])
        # Usamos K=3 vecinos
        self.K = 3 
        
    def _calcular_distancias(self, evento):
        """Calcula la distancia Euclídea normalizada a cada punto de entrenamiento."""
        # Normalizar los datos (simple min-max scaling para la demostración)
        X_min = self.X_train.min(axis=0)
        X_max = self.X_train.max(axis=0)
        
        X_norm = (self.X_train - X_min) / (X_max - X_min)
        evento_norm = (np.array(evento) - X_min) / (X_max - X_min)
        
        # Calcular la distancia euclídea al cuadrado (más rápido)
        distancias_sq = np.sum((X_norm - evento_norm)**2, axis=1)
        return distancias_sq

    def predecir_evento(self, features):
        """Implementación de K-Nearest Neighbors (K-NN) sin scikit-learn."""
        distancias_sq = self._calcular_distancias(features)
        
        # Encontrar los K índices más cercanos
        indices_k_cercanos = np.argsort(distancias_sq)[:self.K]
        
        # Obtener las etiquetas de esos K vecinos
        etiquetas_k_cercanas = self.y_train[indices_k_cercanos]
        
        # Votación: Encontrar la etiqueta que más se repite
        conteo = np.bincount(etiquetas_k_cercanas)
        prediccion = np.argmax(conteo)
        
        return "FUSIÓN BBH (ONDA GRAVITACIONAL) 🌌" if prediccion == 1 else "GLITCH (RUIDO INSTRUMENTAL) 🎧"

# Ejecución inicial para la verificación
if __name__ == "__main__":
    detector = DeepWaveClassifier()
    print("🌟 Iniciando DeepWave - Clasificador de Ondas Gravitacionales")
    print("-------------------------------------------------------")
    print(f"Modelo K-NN ({detector.K} vecinos) listo para clasificar.")
    
    # Evento de prueba 1: Simulación de Fusión (Cerca de [0.1, 50, 0.9])
    evento_real = [0.15, 60, 0.9]
    print(f"\n🔬 Análisis [Amplitud:0.15, Frecuencia:60Hz, Persistencia:0.9]:")
    print(f"  -> Resultado: {detector.predecir_evento(evento_real)}")
    
    # Evento de prueba 2: Simulación de Glitch (Cerca de [0.9, 145, 0.25])
    evento_ruido = [0.95, 140, 0.1]
    print(f"\n🔬 Análisis [Amplitud:0.95, Frecuencia:140Hz, Persistencia:0.1]:")
    print(f"  -> Resultado: {detector.predecir_evento(evento_ruido)}")
    
    # Evento de prueba 3: Punto intermedio (Debería ser Glitch)
    evento_intermedio = [0.5, 100, 0.5]
    print(f"\n🔬 Análisis [Amplitud:0.5, Frecuencia:100Hz, Persistencia:0.5]:")
    print(f"  -> Resultado: {detector.predecir_evento(evento_intermedio)}")
