"""
Features v3: solo las 4 que mostraron correlación real con la etiqueta
en analisis_importancia_features.py, descartando las 4 que eran ruido
(entropía espectral, varianza, pendiente, num_picos).
"""
import numpy as np

def extraer_features_v3(espectrograma):
    n_freq = espectrograma.shape[0]
    corte_baja = min(10, n_freq)
    corte_media = min(30, n_freq)

    energia_baja = np.mean(espectrograma[:corte_baja, :])
    energia_media = np.mean(espectrograma[corte_baja:corte_media, :]) if corte_media > corte_baja else 0.0
    energia_alta = np.mean(espectrograma[corte_media:, :]) if n_freq > corte_media else 0.0
    pico_max = np.max(espectrograma)

    return np.array([energia_baja, energia_media, energia_alta, pico_max])
