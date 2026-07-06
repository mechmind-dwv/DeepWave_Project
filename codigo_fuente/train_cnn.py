"""
Script de entrenamiento de la CNN real para DeepWave, usando el
dataset 100% REAL construido con construir_dataset_real.py (40
eventos GWTC-1/GWTC-2.1 + 80 negativos), NO datos sintéticos.

Ejecutar en un entorno con TensorFlow (Google Colab recomendado,
ya que TensorFlow no instala en Termux/Android).
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from codigo_fuente.deepwave_classifier_cnn_real import RealDeepWaveCNN
from codigo_fuente.deepwave_preprocessing import calcular_espectrograma_stub
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping

FS_REAL = 2048
N_FOLDS = 5
EPOCHS_MAX = 50

def cargar_dataset_real():
    """Carga el dataset real (positivos+negativos) y calcula
    espectrogramas STFT — mismo preprocesamiento que el K-NN."""
    ruta_pos = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_real_positivos.npy")
    ruta_neg = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_real_negativos.npy")

    positivos = np.load(ruta_pos)
    negativos = np.load(ruta_neg)

    X, y = [], []
    for señal in positivos:
        spec = calcular_espectrograma_stub(señal, FS_REAL)
        X.append(spec)
        y.append(1)
    for señal in negativos:
        spec = calcular_espectrograma_stub(señal, FS_REAL)
        X.append(spec)
        y.append(0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    X = np.expand_dims(X, axis=-1)  # canal para Conv2D
    return X, y

def entrenar_con_k_fold(X, y, n_folds=N_FOLDS):
    """Validación k-fold estratificada — con solo 120 muestras, un
    único split train/val dejaría muy pocas muestras de validación
    para confiar en el resultado. K-fold da una estimación más
    honesta del rendimiento real."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    precisiones = []

    print(f"🔬 Validación {n_folds}-fold estratificada sobre {len(y)} muestras reales")
    print("=" * 65)

    for fold, (idx_train, idx_val) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[idx_train], X[idx_val]
        y_train, y_val = y[idx_train], y[idx_val]

        cnn = RealDeepWaveCNN()
        cnn.build_model(input_shape=X.shape[1:])

        early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

        print(f"\n[Fold {fold}/{n_folds}] Entrenamiento: {len(X_train)}, Validación: {len(X_val)}")
        cnn.train(X_train, y_train, X_val, y_val, epochs=EPOCHS_MAX, batch_size=8, callbacks=[early_stop])

        loss, acc = cnn.model.evaluate(X_val, y_val, verbose=0)
        precisiones.append(acc)
        print(f"  Precisión fold {fold}: {acc:.4f}")

    print(f"\n📊 Precisión k-fold promedio: {np.mean(precisiones):.4f} (+/- {np.std(precisiones):.4f})")
    return precisiones

if __name__ == "__main__":
    print("🧠 DeepWave CNN - Entrenamiento con dataset 100% REAL")
    print("=" * 65)

    X, y = cargar_dataset_real()
    print(f"Dataset: {len(y)} muestras ({sum(y)} positivos, {len(y)-sum(y)} negativos)")
    print(f"Forma de cada espectrograma: {X.shape[1:]}")

    precisiones = entrenar_con_k_fold(X, y)

    # Entrenar modelo final con TODOS los datos (para uso en producción)
    print("\n🏁 Entrenando modelo final con el dataset completo...")
    cnn_final = RealDeepWaveCNN()
    cnn_final.build_model(input_shape=X.shape[1:])
    cnn_final.train(X, y, epochs=30, batch_size=8)

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models"), exist_ok=True)
    ruta_modelo = os.path.join(os.path.dirname(__file__), "..", "models", "best_cnn.h5")
    cnn_final.save_model(ruta_modelo)
    print(f"💾 Modelo final guardado en {ruta_modelo}")

    print(f"\n📋 RESUMEN HONESTO:")
    print(f"   Precisión k-fold: {np.mean(precisiones):.1%} (+/- {np.std(precisiones):.1%})")
    print(f"   Comparar con K-NN real (mismo dataset): AUC=0.754, Recall+~67%")
    print(f"   El modelo final se entrenó con TODOS los datos (sin held-out),")
    print(f"   así que su métrica de evaluación no es válida — solo el k-fold lo es.")
