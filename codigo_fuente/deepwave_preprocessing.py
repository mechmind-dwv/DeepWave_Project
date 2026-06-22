import numpy as np

def generar_senal_bbh(duracion_s=1, frecuencia_max=150, tasa_muestreo=2048):
    tiempo = np.linspace(0, duracion_s, int(tasa_muestreo * duracion_s), endpoint=False)
    frecuencia_inicial = 30
    frecuencia_inst = frecuencia_inicial + (frecuencia_max - frecuencia_inicial) * (tiempo / duracion_s)**2
    amplitud = 0.5 * (tiempo / duracion_s) + 0.1
    fase = 2 * np.pi * np.cumsum(frecuencia_inst) / tasa_muestreo
    senal = amplitud * np.sin(fase)
    senal += 0.05 * np.random.normal(size=senal.shape)
    return senal, tasa_muestreo

def generar_senal_glitch(duracion_s=1, tasa_muestreo=2048):
    tiempo = np.linspace(0, duracion_s, int(tasa_muestreo * duracion_s), endpoint=False)
    senal = 0.1 * np.random.normal(size=tiempo.shape)
    inicio_pulso = int(tasa_muestreo * 0.4)
    fin_pulso = int(tasa_muestreo * 0.45)
    senal[inicio_pulso:fin_pulso] += 2.0 * np.random.normal(size=fin_pulso-inicio_pulso)
    return senal, tasa_muestreo

def calcular_espectrograma_stub(senal, tasa_muestreo, ventana_s=0.1, solapamiento_s=0.05):
    puntos_ventana = int(tasa_muestreo * ventana_s)
    puntos_solapamiento = int(tasa_muestreo * solapamiento_s)
    paso = puntos_ventana - puntos_solapamiento
    n_ventanas = int((len(senal) - puntos_ventana) / paso) + 1
    n_freq = puntos_ventana // 2 + 1
    espectrograma_matriz = np.zeros((n_freq, n_ventanas))
    for i in range(n_ventanas):
        inicio = i * paso
        fin = inicio + puntos_ventana
        ventana_datos = senal[inicio:fin]
        fft_resultado = np.fft.rfft(ventana_datos)
        energia = np.abs(fft_resultado)**2
        espectrograma_matriz[:, i] = energia
    espectrograma_log = 10 * np.log10(espectrograma_matriz + 1e-10)
    return espectrograma_log

if __name__ == "__main__":
    print("🧠 DEEPWAVE: Verificación del Módulo de Pre-Procesamiento")
    print("=====================================================")
    senal_bbh, fs = generar_senal_bbh()
    espectrograma_bbh = calcular_espectrograma_stub(senal_bbh, fs)
    print("\n[BBH - Onda Gravitacional 🌌]")
    print(f"Longitud de la señal temporal: {len(senal_bbh)} puntos.")
    print(f"Dimensiones del Espectrograma (Frec. x Tiempo): {espectrograma_bbh.shape}")
    energia_media_bbh = np.mean(espectrograma_bbh[:10, :])
    print(f"Energía media de baja frecuencia (Proxy del Chirp): {energia_media_bbh:.2f} dB")
    senal_glitch, fs = generar_senal_glitch()
    espectrograma_glitch = calcular_espectrograma_stub(senal_glitch, fs)
    print("\n[GLITCH - Ruido Instrumental 🎧]")
    print(f"Longitud de la señal temporal: {len(senal_glitch)} puntos.")
    print(f"Dimensiones del Espectrograma (Frec. x Tiempo): {espectrograma_glitch.shape}")
    energia_max_glitch = np.max(espectrograma_glitch)
    print(f"Pico de Energía Máxima (Proxy del Glitch): {energia_max_glitch:.2f} dB")
    print("\n✅ El Espectrograma ha sido generado y transformado con éxito.")
    print("La matriz resultante (Frecuencia x Tiempo) está lista para una CNN.")
