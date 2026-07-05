"""
Entrena el K-NN usando EXCLUSIVAMENTE el dataset real construido con
construir_dataset_real.py — sin datos sintéticos en ningún paso.
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepwave_preprocessing import calcular_espectrograma_stub
from deepwave_knn_real import extraer_features

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FS_REAL = 2048

class DeepWaveKNNTotalmenteReal:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.X_min = None
        self.X_max = None

    def entrenar_con_datos_reales(self):
        positivos = np.load(os.path.join(DATA_DIR, "dataset_real_positivos.npy"))
        negativos = np.load(os.path.join(DATA_DIR, "dataset_real_negativos.npy"))

        X, y = [], []
        for señal in positivos:
            spec = calcular_espectrograma_stub(señal, FS_REAL)
            X.append(extraer_features(spec))
            y.append(1)
        for señal in negativos:
            spec = calcular_espectrograma_stub(señal, FS_REAL)
            X.append(extraer_features(spec))
            y.append(0)

        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.X_min = self.X_train.min(axis=0)
        self.X_max = self.X_train.max(axis=0)
        print(f"✅ Entrenado con {len(y)} muestras 100% reales ({sum(y)} positivos, {len(y)-sum(y)} negativos)")

    def _normalizar(self, X):
        return (X - self.X_min) / (self.X_max - self.X_min + 1e-10)

    def predecir(self, features):
        X_norm = self._normalizar(self.X_train)
        f_norm = self._normalizar(features)
        distancias = np.sum((X_norm - f_norm) ** 2, axis=1)
        k_cercanos = np.argsort(distancias)[:self.k]
        conteo = np.bincount(self.y_train[k_cercanos], minlength=2)
        prediccion = np.argmax(conteo)
        confianza = conteo[prediccion] / self.k
        return prediccion, confianza

def validacion_leave_one_out(clasificador_cls, k=3):
    """Validación honesta: para cada evento, entrena con TODOS los demás
    y prueba contra el que se dejó fuera. Esto evita el sesgo de
    'probar contra datos que ya viste en entrenamiento'."""
    positivos = np.load(os.path.join(DATA_DIR, "dataset_real_positivos.npy"))
    negativos = np.load(os.path.join(DATA_DIR, "dataset_real_negativos.npy"))

    aciertos, total = 0, 0
    print("\n🔬 VALIDACIÓN LEAVE-ONE-OUT (cada evento probado sin haberlo visto)")
    print("=" * 65)

    todas_señales = list(positivos) + list(negativos)
    todas_etiquetas = [1]*len(positivos) + [0]*len(negativos)

    for i in range(len(todas_señales)):
        X_train, y_train = [], []
        for j in range(len(todas_señales)):
            if j == i:
                continue
            spec = calcular_espectrograma_stub(todas_señales[j], FS_REAL)
            X_train.append(extraer_features(spec))
            y_train.append(todas_etiquetas[j])

        clf = clasificador_cls(k=k)
        clf.X_train = np.array(X_train)
        clf.y_train = np.array(y_train)
        clf.X_min = clf.X_train.min(axis=0)
        clf.X_max = clf.X_train.max(axis=0)

        spec_test = calcular_espectrograma_stub(todas_señales[i], FS_REAL)
        features_test = extraer_features(spec_test)
        pred, conf = clf.predecir(features_test)

        correcto = (pred == todas_etiquetas[i])
        aciertos += correcto
        total += 1
        etiqueta_real = "BBH" if todas_etiquetas[i] == 1 else "RUIDO"
        etiqueta_pred = "BBH" if pred == 1 else "RUIDO"
        marca = "✅" if correcto else "❌"
        print(f"{marca} Muestra {i+1}/{len(todas_señales)}: real={etiqueta_real}, predicho={etiqueta_pred} (conf {conf:.0%})")

    print(f"\n📊 Precisión leave-one-out: {aciertos}/{total} = {aciertos/total:.1%}")
    print("   (Esta es la métrica más honesta que tenemos: cada muestra")
    print("    se probó SIN haber sido vista en su propio entrenamiento)")

if __name__ == "__main__":
    print("🌌 ENTRENAMIENTO K-NN 100% REAL")
    print("=" * 60)

    clf = DeepWaveKNNTotalmenteReal(k=3)
    clf.entrenar_con_datos_reales()

    validacion_leave_one_out(DeepWaveKNNTotalmenteReal, k=3)
