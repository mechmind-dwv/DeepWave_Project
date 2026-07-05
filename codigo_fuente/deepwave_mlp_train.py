#!/usr/bin/env python3
"""
Entrenamiento de un clasificador MLP (red neuronal) para DeepWave.
Usa espectrogramas reales generados con las funciones existentes.
Guarda el modelo en modelo_mlp.joblib.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from joblib import dump

from codigo_fuente.deepwave_preprocessing import (
    generar_senal_bbh,
    generar_senal_glitch,
    calcular_espectrograma_stub
)

# Configuración
N_SAMPLES_PER_CLASS = 500
MODEL_PATH = "modelo_mlp.joblib"
RANDOM_SEED = 42

def generar_dataset():
    X, y = [], []
    print(f"Generando {N_SAMPLES_PER_CLASS} señales BBH...")
    for i in range(N_SAMPLES_PER_CLASS):
        if i % 100 == 0:
            print(f"  BBH {i}/{N_SAMPLES_PER_CLASS}")
        signal, fs = generar_senal_bbh()
        spec = calcular_espectrograma_stub(signal, fs)
        X.append(spec.flatten())  # Aplanar para MLP
        y.append(1)

    print(f"Generando {N_SAMPLES_PER_CLASS} señales Glitch...")
    for i in range(N_SAMPLES_PER_CLASS):
        if i % 100 == 0:
            print(f"  Glitch {i}/{N_SAMPLES_PER_CLASS}")
        signal, fs = generar_senal_glitch()
        spec = calcular_espectrograma_stub(signal, fs)
        X.append(spec.flatten())
        y.append(0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y

def main():
    print("🧠 DeepWave MLP - Entrenamiento")
    print("="*50)

    X, y = generar_dataset()
    print(f"Dataset: {X.shape[0]} muestras, {X.shape[1]} características")

    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Entrenamiento: {len(X_train)}, Validación: {len(X_val)}")

    # MLP con arquitectura similar a la CNN (capas densas)
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        batch_size=32,
        max_iter=100,
        random_state=RANDOM_SEED,
        verbose=True
    )

    print("Entrenando MLP...")
    mlp.fit(X_train, y_train)

    train_acc = mlp.score(X_train, y_train)
    val_acc = mlp.score(X_val, y_val)
    print(f"Precisión entrenamiento: {train_acc:.4f}")
    print(f"Precisión validación: {val_acc:.4f}")

    # Guardar modelo y escalador juntos
    dump({'model': mlp, 'scaler': scaler}, MODEL_PATH)
    print(f"💾 Modelo guardado en {MODEL_PATH}")

if __name__ == "__main__":
    main()
