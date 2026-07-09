"""
Genera el gráfico visual de la curva ROC a partir de los puntos ya
calculados en curva_roc_real.py.
"""
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # sin necesidad de display, genera directo a archivo
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs")

if __name__ == "__main__":
    puntos = np.load(os.path.join(DATA_DIR, "roc_puntos.npy"))
    fpr = puntos[:, 0]
    tpr = puntos[:, 1]

    # AUC por trapecio (recalculado aquí para no depender de otro script)
    auc = np.trapezoid(tpr, fpr)

    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr, color="#FF6B6B", linewidth=2.5, marker="o",
              label=f"K-NN real (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1,
              label="Azar (AUC = 0.500)")

    plt.xlabel("Tasa de Falsos Positivos (FPR)")
    plt.ylabel("Tasa de Verdaderos Positivos (TPR / Recall+)")
    plt.title("Curva ROC — DeepWave K-NN\n(40 eventos reales GWTC-1/2.1, leave-one-out)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ruta_salida = os.path.join(OUTPUT_DIR, "curva_roc_deepwave.png")
    plt.savefig(ruta_salida, dpi=150)
    print(f"💾 Gráfico guardado en {ruta_salida}")
    print(f"📊 AUC confirmado: {auc:.3f}")
