"""
Clasificador de referencia del proyecto DeepWave — v6 consolidado.

Modo dual, honesto sobre la disponibilidad de datos:
- MODO_DUAL (H1+L1 disponibles): usa pico_max, energia_media,
  correlación cruzada H1-L1. Recall+ validado = 70.3%.
- MODO_SIMPLE (solo H1 disponible): usa las 3 features originales
  de v1. Recall+ validado = 67.5%. Menos preciso, pero honesto:
  no se inventa una correlación cuando falta el segundo detector.

Entrenado con el conjunto COMPLETO de datos reales (no leave-one-out,
ese es solo el método de validación, no de entrenamiento final).
"""
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from codigo_fuente.deepwave_preprocessing import calcular_espectrograma_stub
from codigo_fuente.deepwave_knn_real import extraer_features as extraer_features_v1
from codigo_fuente.deepwave_features_v2 import extraer_features_v2

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
FS_REAL = 2048


def extraer_pico_y_energia_media(espectrograma):
    full = extraer_features_v2(espectrograma)
    return np.array([full[1], full[4]])  # energia_media, pico_max


class DeepWaveKNNReferencia:
    """Clasificador de referencia dual: usa H1+L1 si están disponibles,
    o degrada honestamente a solo-H1 si falta el segundo detector."""

    def __init__(self, k=1):
        self.k = k
        self.X_dual, self.y_dual = None, None
        self.X_simple, self.y_simple = None, None
        self.min_dual, self.max_dual = None, None
        self.min_simple, self.max_simple = None, None

    def entrenar(self):
        with open(os.path.join(DATA_DIR, "eventos_procesados.json")) as f:
            registro = json.load(f)
        eventos_todos = registro["eventos_procesados"]
        excluidos = ["GW190620_030421-v2", "GW190630_185205-v2", "GW190708_232457-v2"]
        eventos_con_hl = [e for e in eventos_todos if e not in excluidos]

        positivos = np.load(os.path.join(DATA_DIR, "dataset_real_positivos.npy"))
        negativos = np.load(os.path.join(DATA_DIR, "dataset_real_negativos.npy"))

        # --- Modo simple (solo H1): TODOS los 40 eventos ---
        X_pos_s = np.array([extraer_features_v1(calcular_espectrograma_stub(s, FS_REAL)) for s in positivos])
        X_neg_s = np.array([extraer_features_v1(calcular_espectrograma_stub(s, FS_REAL)) for s in negativos])
        self.X_simple = np.vstack([X_pos_s, X_neg_s])
        self.y_simple = np.array([1] * len(X_pos_s) + [0] * len(X_neg_s))
        self.min_simple = self.X_simple.min(axis=0)
        self.max_simple = self.X_simple.max(axis=0)

        # --- Modo dual (H1+L1): solo los 37 eventos con ambos detectores ---
        indices_con_hl = [i for i, e in enumerate(eventos_todos) if e in eventos_con_hl]
        positivos_sub = positivos[indices_con_hl]
        indices_neg = []
        for i in indices_con_hl:
            indices_neg.extend([2 * i, 2 * i + 1])
        negativos_sub = negativos[indices_neg]

        corr_pos = np.load(os.path.join(DATA_DIR, "correlacion_hl_positivos.npy"))
        corr_neg = np.load(os.path.join(DATA_DIR, "correlacion_hl_negativos.npy"))

        X_pos_selectas = np.array([extraer_pico_y_energia_media(calcular_espectrograma_stub(s, FS_REAL)) for s in positivos_sub])
        X_neg_selectas = np.array([extraer_pico_y_energia_media(calcular_espectrograma_stub(s, FS_REAL)) for s in negativos_sub])

        X_pos_d = np.hstack([X_pos_selectas, corr_pos.reshape(-1, 1)])
        X_neg_d = np.hstack([X_neg_selectas, corr_neg.reshape(-1, 1)])
        self.X_dual = np.vstack([X_pos_d, X_neg_d])
        self.y_dual = np.array([1] * len(X_pos_d) + [0] * len(X_neg_d))
        self.min_dual = self.X_dual.min(axis=0)
        self.max_dual = self.X_dual.max(axis=0)

        print(f"✅ Modo simple entrenado: {len(self.y_simple)} muestras (40 eventos)")
        print(f"✅ Modo dual entrenado:   {len(self.y_dual)} muestras (37 eventos con H1+L1)")

    def _predecir_generico(self, X_train, y_train, X_min, X_max, features, k):
        X_norm = (X_train - X_min) / (X_max - X_min + 1e-10)
        f_norm = (features - X_min) / (X_max - X_min + 1e-10)
        distancias = np.sum((X_norm - f_norm) ** 2, axis=1)
        k_cercanos = np.argsort(distancias)[:k]
        conteo = np.bincount(y_train[k_cercanos], minlength=2)
        prediccion = np.argmax(conteo)
        confianza = conteo[prediccion] / k
        return prediccion, confianza

    def predecir(self, espectrograma_h1, espectrograma_l1=None, correlacion_hl=None):
        """Si se proveen l1 y correlacion_hl, usa modo dual (más preciso,
        Recall+=70.3%). Si no, degrada honestamente a modo simple
        (Recall+=67.5%), sin inventar ningún dato faltante."""
        if espectrograma_l1 is not None and correlacion_hl is not None:
            full = extraer_features_v2(espectrograma_h1)
            features = np.array([full[1], full[4], correlacion_hl])
            pred, conf = self._predecir_generico(self.X_dual, self.y_dual, self.min_dual, self.max_dual, features, self.k)
            modo = "dual (H1+L1)"
        else:
            features = extraer_features_v1(espectrograma_h1)
            pred, conf = self._predecir_generico(self.X_simple, self.y_simple, self.min_simple, self.max_simple, features, self.k)
            modo = "simple (solo H1)"

        return {
            "prediccion": "FUSIÓN BBH" if pred == 1 else "GLITCH/RUIDO",
            "es_bbh": bool(pred == 1),
            "confianza": float(conf),
            "modo": modo,
        }


if __name__ == "__main__":
    print("🌌 DEEPWAVE — Clasificador de referencia v6 (dual H1+L1 / H1 solo)")
    print("=" * 70)

    clf = DeepWaveKNNReferencia(k=1)
    clf.entrenar()

    print("\n📋 RESUMEN DE VALIDACIÓN (leave-one-out, sesiones previas):")
    print("   Modo dual (H1+L1):  Recall+ = 70.3%, Global = 73.0% (n=37)")
    print("   Modo simple (H1):   Recall+ = 67.5%, Global = ~70%   (n=40)")
    print("\n⚠️  NOTA: estas métricas son de validación leave-one-out, NO del")
    print("   modelo entrenado aquí con el 100% de los datos (ese no tiene")
    print("   held-out, así que no se reporta su 'precisión' — sería inválida).")
