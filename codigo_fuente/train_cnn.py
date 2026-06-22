"""
Script de entrenamiento de la CNN real para DeepWave.
Ejecutar en un entorno con TensorFlow (PC o Google Colab).
"""
import numpy as np
import os
import sys

# Asegurar que podemos importar desde codigo_fuente
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from codigo_fuente.deepwave_classifier_cnn_real import RealDeepWaveCNN
from codigo_fuente.deepwave_preprocessing import (
    generar_senal_bbh, generar_senal_glitch, calcular_espectrograma_stub
)
from sklearn.model_selection import train_test_split

def generar_dataset(n_samples=5000):
    """Genera espectrogramas balanceados BBH/Glitch usando las funciones de preprocesamiento."""
    X, y = [], []
    for i in range(n_samples // 2):
        # BBH
        señal, fs = generar_senal_bbh()
        spec = calcular_espectrograma_stub(señal, fs)
        X.append(spec)
        y.append(1)
        # Glitch
        señal, fs = generar_senal_glitch()
        spec = calcular_espectrograma_stub(señal, fs)
        X.append(spec)
        y.append(0)
        if (i+1) % 500 == 0:
            print(f"Generadas {i+1} muestras...")
    X = np.array(X)
    y = np.array(y)
    # Añadir canal (1)
    X = np.expand_dims(X, axis=-1)
    return X, y

if __name__ == "__main__":
    print("Generando dataset de entrenamiento (sintético, pero con STFT real)...")
    X, y = generar_dataset(2000)   # Usa más muestras en PC
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cnn = RealDeepWaveCNN()
    cnn.build_model()
    print("Entrenando CNN real...")
    history = cnn.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=32)

    # Guardar modelo
    os.makedirs("models", exist_ok=True)
    cnn.save_model("models/best_cnn.h5")
    print("Modelo guardado en models/best_cnn.h5")

    # Evaluación final
    loss, acc = cnn.model.evaluate(X_val, y_val, verbose=0)
    print(f"Precisión en validación: {acc:.4f}")
