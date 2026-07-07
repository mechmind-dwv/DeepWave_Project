# 📋 DeepWave — Pendientes (actualizado 5 julio 2026)

## 🟡 Prioridad media

- [x] ~~Ampliar la validación a más eventos reales de GWOSC~~ — **SUPERADO**:
      se amplió a 40 eventos reales (GWTC-1 completo + 29 de GWTC-2.1),
      no solo GW170814/GW190521. `construir_dataset_real.py` generaliza
      la extracción de negativos por GPS time para cualquier evento.
      Ver rama `feature/mlp-classifier`.
- [x] ~~Curva ROC/AUC real~~ — **COMPLETADO**: AUC=0.754 sobre 40
      eventos reales (leave-one-out, score continuo K=15). Gráfico en
      `docs/curva_roc_deepwave.png`. Punto de operación óptimo
      identificado (umbral 0.4 → TPR=60%, FPR=7.5%).
- [x] ~~Entrenar la CNN real en Google Colab~~ — **COMPLETADO, resultado
      negativo honesto**: k-fold da 67.5%±1.7%, pero la matriz de
      confusión revela Recall+=25% (vs 67.5% del K-NN). Con solo 120
      muestras, la CNN no supera al K-NN con features artesanales.
      Documentado en DIVULGACION_PERSEO.md. NO se implementó inferencia
      en Termux (tflite-runtime no tiene wheel; no vale la pena
      reconstruir el forward-pass a mano para un modelo que no gana).

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

- [x] Dashboard conectado al clasificador K-NN real: ya no modo demo
      aleatorio, sino clasificación real vía espectrograma STFT de una
      señal generada a partir de los parámetros del usuario. Gráficos
      de forma de onda y espectrograma también muestran la señal real
      del último análisis, no una réplica cosmética.

- [x] Divulgación Perseo↔DeepWave documentada y fusionada a `main`
- [x] SSH configurado correctamente (`~/.ssh/config`)
- [x] Whitening real (PSD Welch + Butterworth) implementado y validado
- [x] K-NN nativo sin TensorFlow, funcional en Termux
- [x] Validación con control negativo (1 positivo real + 2 negativos reales)
- [x] 3 fixes de CodeRabbit aplicados (reset, POST en test_sequence, límite de historial)
- [x] Manual de usuario reescrito sin métricas inventadas

## 🔬 Rama feature/mlp-classifier (5-6 julio 2026)

- [x] MLP con espectrograma completo: **descartado**, falló control
      negativo real (documentado en DIVULGACION_PERSEO.md)
- [x] Dataset 100% real construido: 40 eventos GWTC-1/GWTC-2.1 +
      80 negativos, sin síntesis (GW190425 excluido por 34% NaN)
- [x] K-NN entrenado con datos 100% reales, validación leave-one-out
      en 3 escalas (11→25→40 eventos)
- [x] Hallazgo documentado: meseta de Recall+ (~68%) al pasar de 25
      a 40 eventos — el cuello de botella pasó de "faltan datos" a
      "faltan mejores features"

### Próximos pasos concretos para esta rama
- [ ] Enriquecer features: probar añadir más estadísticos del
      espectrograma (varianza, entropía espectral, número de picos)
      en vez de solo las 3 actuales (energía baja, pendiente, pico)
- [ ] Considerar coeficientes wavelet como alternativa a STFT simple
- [ ] Si las features mejoradas no rompen la meseta, sería evidencia
      de que hace falta un modelo más expresivo (red neuronal simple
      bien regularizada, no el MLP que sobreajustó) entrenado con
      más datos (80-100 eventos de GWTC-3 completo)
- [ ] Decidir si esta rama se fusiona a `desarrollo` tal cual (como
      "K-NN validado con datos reales, con limitaciones documentadas")
      o si espera a la siguiente iteración de features
