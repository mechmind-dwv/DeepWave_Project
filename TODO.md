# 📋 DeepWave — Pendientes (actualizado 5 julio 2026)

## 🔴 Prioridad alta

- [ ] **Conectar `dashboard.py` al clasificador real.** Actualmente los
      imports (`WaveAnalyzer`, `generate_spectrogram`) no existen en el
      código real, así que el dashboard siempre corre en modo demo con
      resultados aleatorios. Cambiar a `DeepWaveKNNReal` +
      `calcular_espectrograma_stub` (ver `codigo_fuente/deepwave_knn_real.py`).

## 🟡 Prioridad media

- [ ] **Ampliar la validación a más eventos reales de GWOSC**
      (GW170814, GW190521) para tener más de N=1 positivo. Generalizar
      `deepwave_control_negativo.py` para aceptar cualquier evento por
      GPS time en vez de tenerlo hardcodeado a GW150914.
- [ ] **Curva ROC/AUC real** una vez haya más de un evento — daría una
      métrica honesta de tasa de falsos positivos, algo que el K-NN
      actual con K=5 y confianza binaria no ofrece.
- [ ] **Entrenar la CNN real** (`deepwave_classifier_cnn_real.py` +
      `train_cnn.py`) en Google Colab, ya que TensorFlow no instala en
      Termux/Android. Descargar el `.h5` resultante y decidir cómo
      hacer inferencia en el dispositivo (reimplementar forward-pass a
      mano con numpy, o usar tflite-runtime si hay wheel disponible).

## 🟢 Prioridad baja / limpieza

- [ ] Revisar el commit duplicado `0d4d7d0` / `3407589` en el historial
      de `desarrollo` (mismo mensaje, sin impacto funcional, no urge).
- [ ] Confirmar si aún hace falta `deepwave_datos_reales.py` (whitening
      simplificado) ahora que existe `deepwave_whitening_real.py`
      (whitening real por Welch) — posible duplicado a limpiar.
- [ ] Considerar mover el token de `~/.git-credentials` a algo más
      seguro que texto plano, o confirmar su fecha de expiración (90
      días desde ~5 julio 2026).

## ✅ Completado (5 julio 2026)

- [x] Divulgación Perseo↔DeepWave documentada y fusionada a `main`
- [x] SSH configurado correctamente (`~/.ssh/config`)
- [x] Whitening real (PSD Welch + Butterworth) implementado y validado
- [x] K-NN nativo sin TensorFlow, funcional en Termux
- [x] Validación con control negativo (1 positivo real + 2 negativos reales)
- [x] 3 fixes de CodeRabbit aplicados (reset, POST en test_sequence, límite de historial)
- [x] Manual de usuario reescrito sin métricas inventadas
