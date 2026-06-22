import numpy as np
from codigo_fuente.deepwave_preprocessing import generar_senal_bbh, generar_senal_glitch, calcular_espectrograma_stub

class DeepWaveCNN:
    def __init__(self, input_shape=(103, 19)):
        self.input_shape = input_shape
        self.filtros_conv1 = 16
        self.kernel_conv1 = (3, 3)
        self.pool_size1 = (2, 2)
        self.filtros_conv2 = 32
        self.kernel_conv2 = (3, 3)
        self.salida_clases = 2

    def simular_capa_conv(self, matriz_entrada, filtros, kernel_shape):
        sim_alto = matriz_entrada.shape[0] - kernel_shape[0] + 1
        sim_ancho = matriz_entrada.shape[1] - kernel_shape[1] + 1
        output_shape = (sim_alto, sim_ancho, filtros)
        print(f"  -> Conv: {matriz_entrada.shape} -> {output_shape} (Kernel {kernel_shape})")
        return np.random.rand(*output_shape)

    def simular_capa_pooling(self, matriz_entrada, pool_size):
        sim_alto = matriz_entrada.shape[0] // pool_size[0]
        sim_ancho = matriz_entrada.shape[1] // pool_size[1]
        sim_canales = matriz_entrada.shape[2]
        output_shape = (sim_alto, sim_ancho, sim_canales)
        print(f"  -> Pool: {matriz_entrada.shape} -> {output_shape} (Pool {pool_size})")
        return np.random.rand(*output_shape)

    def simular_forward_pass(self, espectrograma_log):
        X = np.expand_dims(espectrograma_log, axis=-1)
        print(f"Entrada (Espectrograma): {X.shape}")
        X = self.simular_capa_conv(X, self.filtros_conv1, self.kernel_conv1)
        X = self.simular_capa_pooling(X, self.pool_size1)
        X = self.simular_capa_conv(X, self.filtros_conv2, self.kernel_conv2)
        n_features_final = X.size
        print(f"  -> Aplanado (Flatten): {n_features_final} características")
        probabilidad_bbh = np.random.rand()
        clase_predicha = 1 if probabilidad_bbh > 0.5 else 0
        return clase_predicha, probabilidad_bbh

if __name__ == "__main__":
    detector_cnn = DeepWaveCNN()
    print("🧠 DEEPWAVE: Verificación de la Arquitectura CNN")
    print("================================================")
    senal_bbh, fs = generar_senal_bbh()
    espectrograma_bbh = calcular_espectrograma_stub(senal_bbh, fs)
    print("\n--- Simulación de Predicción CNN (BBH) ---")
    clase, prob = detector_cnn.simular_forward_pass(espectrograma_bbh)
    resultado_texto = "FUSIÓN BBH 🌌" if clase == 1 else "GLITCH 🎧"
    print(f"\n✅ Clasificación Final Simulación: {resultado_texto}")
    print(f"Probabilidad simulada de BBH: {prob:.4f}")
    print("\n================================================")
    print("La arquitectura ha sido definida y las dimensiones de la matriz son válidas para el flujo de la CNN.")
