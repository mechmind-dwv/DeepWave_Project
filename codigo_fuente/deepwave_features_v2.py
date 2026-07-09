"""
Features enriquecidas del espectrograma STFT — versión 2.
En vez de 3 estadísticos (energía baja, pendiente, pico), extrae 8:
más resolución en banda de frecuencia, entropía espectral, varianza,
y conteo de picos temporales. Sin dependencias nuevas (solo numpy).
"""
import numpy as np

def extraer_features_v2(espectrograma):
    """
    espectrograma: matriz (n_freq, n_tiempo) en dB, salida de
    calcular_espectrograma_stub().

    Devuelve un vector de 8 features:
    1. energia_baja    - energía media en las 10 primeras filas de frecuencia
    2. energia_media    - energía media en filas 10-30 (banda intermedia)
    3. energia_alta     - energía media en filas 30+ (banda alta)
    4. pendiente        - tendencia temporal de la energía total (¿sube o baja?)
    5. pico_max         - pico máximo de energía en toda la matriz
    6. varianza_total    - dispersión general de la energía (qué tan "ruidosa" es la forma)
    7. entropia_espectral - entropía de Shannon de la distribución de energía
                           (una señal muy concentrada en pocas frecuencias
                           tiene entropía baja; ruido uniforme tiene entropía alta)
    8. num_picos        - cantidad de picos locales en la curva de energía
                           por tiempo (proxy de "cuántas veces sube y baja")
    """
    n_freq = espectrograma.shape[0]
    corte_baja = min(10, n_freq)
    corte_media = min(30, n_freq)

    energia_baja = np.mean(espectrograma[:corte_baja, :])
    energia_media = np.mean(espectrograma[corte_baja:corte_media, :]) if corte_media > corte_baja else 0.0
    energia_alta = np.mean(espectrograma[corte_media:, :]) if n_freq > corte_media else 0.0

    energia_por_tiempo = np.mean(espectrograma, axis=0)
    pendiente = np.polyfit(np.arange(len(energia_por_tiempo)), energia_por_tiempo, 1)[0]

    pico_max = np.max(espectrograma)
    varianza_total = np.var(espectrograma)

    # Entropía espectral: normalizar la energía (desplazada a positivo) a
    # una distribución de probabilidad, y calcular -sum(p*log(p))
    energia_positiva = espectrograma - espectrograma.min() + 1e-10
    p = energia_positiva.flatten() / energia_positiva.sum()
    entropia_espectral = -np.sum(p * np.log(p + 1e-10))

    # Número de picos locales en la curva de energía temporal
    diffs = np.diff(energia_por_tiempo)
    signos = np.sign(diffs)
    cambios_signo = np.diff(signos)
    num_picos = np.sum(cambios_signo < 0)  # de subida a bajada = pico

    return np.array([
        energia_baja, energia_media, energia_alta, pendiente,
        pico_max, varianza_total, entropia_espectral, float(num_picos)
    ])

if __name__ == "__main__":
    # Verificación rápida con una señal sintética
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from deepwave_preprocessing import generar_senal_bbh, generar_senal_glitch, calcular_espectrograma_stub

    señal_bbh, fs = generar_senal_bbh()
    spec_bbh = calcular_espectrograma_stub(señal_bbh, fs)
    features_bbh = extraer_features_v2(spec_bbh)
    print("Features v2 (BBH sintético):", features_bbh)

    señal_glitch, fs = generar_senal_glitch()
    spec_glitch = calcular_espectrograma_stub(señal_glitch, fs)
    features_glitch = extraer_features_v2(spec_glitch)
    print("Features v2 (Glitch sintético):", features_glitch)
