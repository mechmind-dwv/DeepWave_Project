"""
MÓDULO DEEPWAVE PRE-PROCESAMIENTO - Generación de Espectrogramas
Autores: Benjamin Cabeza Duran / Gemini IA
Fecha: Octubre 2025
"""

import numpy as np

# --- 1. Generación de Señales (Simulación) ---

def generar_senal_bbh(duracion_s=1, frecuencia_max=150, tasa_muestreo=2048):
    """Simula una señal de 'chirp' de Fusión BBH."""
    tiempo = np.linspace(0, duracion_s, int(tasa_muestreo * duracion_s), endpoint=False)
    
    # La frecuencia aumenta con el tiempo (efecto chirp)
    frecuencia_inicial = 30
    frecuencia_inst = frecuencia_inicial + (frecuencia_max - frecuencia_inicial) * (tiempo / duracion_s)**2
    
    # Señal: Amplitud modulada * sin(fase)
    # La amplitud aumenta al final (donde ocurre la fusión)
    amplitud = 0.5 * (tiempo / duracion_s) + 0.1
    fase = 2 * np.pi * np.cumsum(frecuencia_inst) / tasa_muestreo
    
    senal = amplitud * np.sin(fase)
    # Añadir un poco de ruido de fondo
    senal += 0.05 * np.random.normal(size=senal.shape)
    
    return senal, tasa_muestreo

def generar_senal_glitch(duracion_s=1, tasa_muestreo=2048):
    """Simula un 'glitch' (ruido impulsivo breve y de banda ancha)."""
    tiempo = np.linspace(0, duracion_s, int(tasa_muestreo * duracion_s), endpoint=False)
    senal = 0.1 * np.random.normal(size=tiempo.shape) # Ruido de fondo
    
    # Añadir un pulso de ruido aleatorio (glitch)
    inicio_pulso = int(tasa_muestreo * 0.4)
    fin_pulso = int(tasa_muestreo * 0.45)
    senal[inicio_pulso:fin_pulso] += 2.0 * np.random.normal(size=fin_pulso-inicio_pulso)
    
    return senal, tasa_muestreo

# --- 2. Pre-Procesamiento (Simulación STFT/Espectrograma) ---

def calcular_espectrograma_stub(senal, tasa_muestreo, ventana_s=0.1, solapamiento_s=0.05):
    """
    Simula el cálculo de un Espectrograma (STFT) con NumPy.
    Transforma el dominio de tiempo a tiempo-frecuencia.
    """
    
    puntos_ventana = int(tasa_muestreo * ventana_s)
    puntos_solapamiento = int(tasa_muestreo * solapamiento_s)
    paso = puntos_ventana - puntos_solapamiento
    
    n_ventanas = int((len(senal) - puntos_ventana) / paso) + 1
    
    # Inicializar la matriz del espectrograma
    # La dimensión de frecuencia es la mitad de la ventana + 1 (por simetría FFT)
    n_freq = puntos_ventana // 2 + 1
    espectrograma_matriz = np.zeros((n_freq, n_ventanas))
    
    for i in range(n_ventanas):
        inicio = i * paso
        fin = inicio + puntos_ventana
        ventana_datos = senal[inicio:fin]
        
        # Aplicar la Transformada Rápida de Fourier (FFT)
        fft_resultado = np.fft.rfft(ventana_datos)
        
        # Energía (magnitud al cuadrado)
        energia = np.abs(fft_resultado)**2
        
        # Almacenar la energía en la columna del espectrograma
        espectrograma_matriz[:, i] = energia
        
    # Normalizar logarítmicamente para realzar el contraste (como se haría en la práctica)
    espectrograma_log = 10 * np.log10(espectrograma_matriz + 1e-10) # +1e-10 para evitar log(0)
    
    return espectrograma_log

# --- 3. Verificación de la Transformación ---

if __name__ == "__main__":
    print("🧠 DEEPWAVE: Verificación del Módulo de Pre-Procesamiento")
    print("=====================================================")
    
    # a) Generar señal de Fusión BBH
    senal_bbh, fs = generar_senal_bbh()
    espectrograma_bbh = calcular_espectrograma_stub(senal_bbh, fs)
    
    print("\n[BBH - Onda Gravitacional 🌌]")
    print(f"Longitud de la señal temporal: {len(senal_bbh)} puntos.")
    print(f"Dimensiones del Espectrograma (Frec. x Tiempo): {espectrograma_bbh.shape}")
    
    # Un BBH 'chirp' debería tener un patrón ascendente de energía
    # Verificamos la energía promedio en la parte de baja frecuencia (índices bajos)
    energia_media_bbh = np.mean(espectrograma_bbh[:10, :])
    print(f"Energía media de baja frecuencia (Proxy del Chirp): {energia_media_bbh:.2f} dB")
    
    # b) Generar señal de Glitch
    senal_glitch, fs = generar_senal_glitch()
    espectrograma_glitch = calcular_espectrograma_stub(senal_glitch, fs)
    
    print("\n[GLITCH - Ruido Instrumental 🎧]")
    print(f"Longitud de la señal temporal: {len(senal_glitch)} puntos.")
    print(f"Dimensiones del Espectrograma (Frec. x Tiempo): {espectrograma_glitch.shape}")
    
    # Un Glitch debería tener energía dispersa en todas las frecuencias en un instante
    # Verificamos la energía máxima para ver el pico del glitch
    energia_max_glitch = np.max(espectrograma_glitch)
    print(f"Pico de Energía Máxima (Proxy del Glitch): {energia_max_glitch:.2f} dB")
    
    print("\n✅ El Espectrograma ha sido generado y transformado con éxito.")
    print("La matriz resultante (Frecuencia x Tiempo) está lista para una CNN.")
