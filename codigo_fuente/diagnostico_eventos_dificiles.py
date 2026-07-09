"""
Identifica qué eventos específicos falla el K-NN consistentemente,
para confirmar si coincide con los de SNR bajo conocido en la
literatura de LIGO (GW151012, GW170729, entre otros).
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepwave_preprocessing import calcular_espectrograma_stub
from deepwave_knn_real import extraer_features
from experimentar_knn_real import cargar_features_reales, knn_predecir

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

EVENTOS = ["GW150914", "GW151012", "GW151226", "GW170104", "GW170608",
           "GW170729", "GW170809", "GW170814", "GW170817", "GW170818", "GW170823"]

if __name__ == "__main__":
    X_pos, X_neg = cargar_features_reales()

    print("🔬 DIAGNÓSTICO: ¿qué eventos falla el K-NN consistentemente?")
    print("=" * 65)
    print(f"{'Evento':<12} {'K=1':>6} {'K=3':>6} {'K=5':>6}  Features (energía_baja, pendiente, pico)")
    print("-" * 65)

    for i, evento in enumerate(EVENTOS):
        X_train_base = np.vstack([np.delete(X_pos, i, axis=0), X_neg])
        y_train_base = np.array([1]*(len(X_pos)-1) + [0]*len(X_neg))
        features_test = X_pos[i]

        resultados = []
        for k in [1, 3, 5]:
            pred, _ = knn_predecir(X_train_base, y_train_base, features_test, k)
            resultados.append("✅BBH" if pred == 1 else "❌ruido")

        print(f"{evento:<12} {resultados[0]:>6} {resultados[1]:>6} {resultados[2]:>6}  {features_test}")

    print("\n📋 Eventos consistentemente mal clasificados (❌ en las 3 columnas)")
    print("   son candidatos a tener SNR real bajo, no un bug del pipeline.")
