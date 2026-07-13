# 📋 DeepWave — Pendientes (actualizado 12 julio 2026)

## 🏆 CAMPAÑA COMPLETADA: Dataset de 350 eventos reales

- [x] **Objetivo alcanzado:** 350 eventos positivos + 700 negativos,
      100% datos reales de LIGO/Virgo (GWOSC), sin ninguna síntesis.
      Catálogos usados: GWTC-1, GWTC-2, GWTC-2.1, GWTC-3, GWTC-4.0,
      GWTC-4.1, GWTC-5.0, O3_Discovery_Papers, O4_Discovery_Papers,
      IAS-O3a, GWTC-1/2.1/3-marginal.
- [x] Sistema de versionado de datasets (`data/versions/dataset_v_n*/`)
      con 8 snapshots reproducibles: n=107, 116, 133, 151, 201, 214,
      290, 350.
- [x] Guardado incremental cada 5 eventos en `construir_dataset_real.py`
      — corrección crítica tras detectar pérdida de progreso en
      tandas grandes sin protección parcial.
- [x] `descargar_evento()` en `construir_dataset_l1.py` maneja
      detector ausente con `FileNotFoundError` explícito.

### Resultados finales de la campaña

| Métrica | Valor |
|---|---|
| AUC agregado (bootstrap, 2000 remuestreos) | 0.735 (IC95%: 0.705–0.764) |
| AUC condicionado SNR alto (n=176) | 0.797 |
| AUC condicionado SNR bajo (n=172) | 0.662 |
| Precisión leave-one-out global | 68.5% (719/1050) |

**Curva completa de AUC condicionado por SNR (8 puntos):**
107→0.757/0.590 · 116→0.765/0.602 · 133→0.788/0.602 · 151→0.793/0.639
· 201→0.772/0.637 · 214→0.771/0.610 · 290→0.794/0.679 · 350→0.797/0.662

**Conclusión:** la estructura de dos regímenes (SNR alto≈0.80,
SNR bajo≈0.66) es estable y reproducible. Este es el techo real del
método actual (K-NN, 3 features espectrales, un solo detector H1).

## 🎯 Próximos pasos (decisión pendiente)

- [ ] **Decidir dirección:** ¿fusionar esta rama a `desarrollo` ahora
      como release consolidado, o intentar el salto metodológico
      (matched filtering, o CNN reentrenada con 350 eventos en vez
      de los 120 con los que falló antes) antes de fusionar?
- [ ] Si se reintenta la CNN: ahora hay ~3x más datos que cuando
      falló (Recall+=25% con n=120) — vale la pena repetir el
      experimento en Colab con el dataset completo de 350.
- [ ] Recalcular correlación H1-L1 (v6) sobre el dataset completo de
      350 — la última vez (n=68) v1≈v6, pero con casi 5x más datos
      podría cambiar la conclusión.
- [ ] Considerar publicar el proyecto/hallazgos en algún formato
      compartible (blog post, paper corto, o simplemente el repo
      bien documentado tal cual está).

## 🟡 Infraestructura y limpieza pendiente (sin urgencia)

- [ ] Migrar `dataset_real_*.npy` a Git LFS/DVC (sugerencia CodeRabbit)
      — evaluar ahora que la campaña de descarga terminó.
- [ ] Arreglar el import de `deepwave_knn_real.py`: actualmente solo
      funciona como paquete (`from codigo_fuente.X import`), rompe la
      ejecución directa de scripts como `entrenar_knn_real.py` desde
      dentro de `codigo_fuente/`. Usar try/except para soportar ambos
      contextos.
- [ ] Mejoras pendientes en `construir_dataset_l1.py`: guardado
      incremental cada 10 eventos, informe JSON de eventos omitidos
      con causa, barra de progreso.
- [ ] Revisar commit duplicado `0d4d7d0`/`3407589` en `desarrollo`.
- [ ] Confirmar si `deepwave_datos_reales.py` es redundante con
      `deepwave_whitening_real.py`.
- [ ] Revisar expiración del token en `~/.git-credentials`.
- [ ] Conectar `dashboard.py` al clasificador entrenado con 350
      eventos (actualmente usa el K-NN con datos sintéticos para la
      demo interactiva — separar claramente "demo con sliders" de
      "resultado científico con datos reales" en la interfaz).

## 🔬 Historial de hallazgos clave (resumen)

- [x] MLP con espectrograma completo: descartado (falló control
      negativo real)
- [x] Features v2/v3 (más features, o seleccionadas por correlación
      individual): no superaron a v1 con datasets pequeños
- [x] v6 (features selectas + correlación H1-L1): mejor con n=37,
      pero convergió con v1 al ampliar a n=68 — pendiente reintentar
      con n=350
- [x] CNN (Colab, n=120): Recall+=25%, muy por debajo del K-NN —
      pendiente reintentar con n=350
- [x] Descubrimiento del efecto SNR: el AUC agregado dependía
      fuertemente de la composición SNR de la muestra, no era una
      propiedad estable del modelo — resuelto reportando AUC
      condicionado por SNR

## ✅ Hitos históricos completados

- [x] Divulgación Perseo↔DeepWave, fusionada a `main`
- [x] Whitening real (Welch + Butterworth), validado contra GW150914
- [x] K-NN nativo sin TensorFlow, funcional en Termux
- [x] Dashboard conectado al K-NN real (no modo demo aleatorio)
- [x] Revisión metodológica externa incorporada (matices de alcance,
      lenguaje técnico en README, autoría transparente)
- [x] **Campaña de 350 eventos reales completada (12 julio 2026)**
