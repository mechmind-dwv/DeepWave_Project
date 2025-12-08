"""
MÓDULO DEEPWAVE CLASIFICADOR CNN - Modelo Arquitectónico Simulado
Autores: Benjamin Cabeza Duran / Gemini IA
Fecha: Octubre 2025
"""

import numpy as np
from deepwave_preprocessing import generar_senal_bbh, generar_senal_glitch, calcular_espectrograma_stub

class DeepWaveCNN:
    def __init__(self, input_shape=(103, 19)):
        # La forma de entrada es (Frecuencia, Tiempo) del espectrograma
        self.input_shape = input_shape
        
        # --- Parámetros de la Arquitectura CNN (Simulados) ---
        # 1. Capa Convolucional (Conv1)
        self.filtros_conv1 = 16
        self.kernel_conv1 = (3, 3) # Un "ojo" de 3x3 para buscar patrones pequeños
        
        # 2. Capa de Max Pooling (Pool1)
        self.pool_size1 = (2, 2) # Reduce a la mitad las dimensiones

        # 3. Capa Convolucional (Conv2)
        self.filtros_conv2 = 32
        self.kernel_conv2 = (3, 3) # Un "ojo" más grande
        
        # 4. Capa Densa Final (Clasificación)
        self.salida_clases = 2 # BBH (1) o GLITCH (0)
        
    def simular_capa_conv(self, matriz_entrada, filtros, kernel_shape):
        """Simula la operación de convolución (extracción de rasgos)."""
        # Una convolución real es compleja. Aquí simulamos la reducción de dimensión 
        # y la 'detección de bordes' promediando áreas locales.
        
        # La nueva forma es (Alto - Kernel + 1) x (Ancho - Kernel + 1)
        # Asumimos una reducción de 2 unidades por dimensión para un kernel 3x3 (sin padding)
        sim_alto = matriz_entrada.shape[0] - kernel_shape[0] + 1
        sim_ancho = matriz_entrada.shape[1] - kernel_shape[1] + 1
        
        # El resultado de la convolución será una matriz con la nueva forma, 
        # multiplicada por el número de filtros (canales)
        output_shape = (sim_alto, sim_ancho, filtros)
        
        print(f"  -> Conv: {matriz_entrada.shape} -> {output_shape} (Kernel {kernel_shape})")
        return np.random.rand(*output_shape) # Devuelve una matriz de dimensiones correctas

    def simular_capa_pooling(self, matriz_entrada, pool_size):
        """Simula la operación de Max Pooling (reducción de ruido y dimensión)."""
        # Max Pooling reduce las dimensiones por el factor pool_size
        sim_alto = matriz_entrada.shape[0] // pool_size[0]
        sim_ancho = matriz_entrada.shape[1] // pool_size[1]
        sim_canales = matriz_entrada.shape[2]
        
        output_shape = (sim_alto, sim_ancho, sim_canales)
        
        print(f"  -> Pool: {matriz_entrada.shape} -> {output_shape} (Pool {pool_size})")
        return np.random.rand(*output_shape)
        
    def simular_forward_pass(self, espectrograma_log):
        """Simula el flujo de datos a través de la arquitectura CNN."""
        
        # 1. Entrada (Espectrograma)
        X = np.expand_dims(espectrograma_log, axis=-1) # Añadir dimensión de 'canales' (103, 19, 1)
        print(f"Entrada (Espectrograma): {X.shape}")
        
        # 2. Capa Conv1 + ReLU (Activación)
        X = self.simular_capa_conv(X, self.filtros_conv1, self.kernel_conv1)
        
        # 3. Capa Pool1
        X = self.simular_capa_pooling(X, self.pool_size1)
        
        # 4. Capa Conv2 + ReLU
        X = self.simular_capa_conv(X, self.filtros_conv2, self.kernel_conv2)
        
        # 5. Capa Densa (Clasificación Final)
        # Se aplana el resultado (Flatten) para la capa densa
        n_features_final = X.size
        print(f"  -> Aplanado (Flatten): {n_features_final} características")
        
        # La predicción final es una probabilidad simulada (0.0 a 1.0)
        probabilidad_bbh = np.random.rand()
        clase_predicha = 1 if probabilidad_bbh > 0.5 else 0
        
        return clase_predicha, probabilidad_bbh

# --- 5. Verificación de la Arquitectura ---

if __name__ == "__main__":
    detector_cnn = DeepWaveCNN()
    
    print("🧠 DEEPWAVE: Verificación de la Arquitectura CNN")
    print("================================================")
    
    # 1. Pre-procesar una señal de prueba (BBH)
    senal_bbh, fs = generar_senal_bbh()
    espectrograma_bbh = calcular_espectrograma_stub(senal_bbh, fs)
    
    print("\n--- Simulación de Predicción CNN (BBH) ---")
    clase, prob = detector_cnn.simular_forward_pass(espectrograma_bbh)
    
    resultado_texto = "FUSIÓN BBH 🌌" if clase == 1 else "GLITCH 🎧"
    print(f"\n✅ Clasificación Final Simulación: {resultado_texto}")
    print(f"Probabilidad simulada de BBH: {prob:.4f}")

    print("\n================================================")
    print("La arquitectura ha sido definida y las dimensiones de la matriz son válidas para el flujo de la CNN.")
