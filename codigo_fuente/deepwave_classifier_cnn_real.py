"""
MÓDULO DEEPWAVE CLASIFICADOR CNN REAL (Keras/TensorFlow)
Autor: DeepWave Team / MechMind
Fecha: Junio 2026
"""
import numpy as np
import os

class RealDeepWaveCNN:
    """CNN real para clasificación de espectrogramas GW (BBH vs Glitch)."""

    def __init__(self, model_path=None):
        self.model = None
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def build_model(self, input_shape=(103, 19, 1)):
        try:
            from tensorflow.keras import layers, models
        except ImportError:
            raise ImportError(
                "TensorFlow no está instalado. Instálalo con: pip install tensorflow"
            )

        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model = model
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=30, batch_size=32):
        if self.model is None:
            self.build_model()
        validation_data = (X_val, y_val) if X_val is not None else None
        history = self.model.fit(X_train, y_train,
                                 validation_data=validation_data,
                                 epochs=epochs,
                                 batch_size=batch_size)
        return history

    def predict(self, espectrograma):
        """Predice: 1=BBH, 0=Glitch. Devuelve (clase, probabilidad)."""
        if self.model is None:
            raise RuntimeError("El modelo no está entrenado.")
        # espectrograma: (103, 19) -> (1, 103, 19, 1)
        X = np.expand_dims(np.expand_dims(espectrograma, axis=0), axis=-1)
        prob = self.model.predict(X, verbose=0)[0][0]
        return (1 if prob > 0.5 else 0), prob

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
