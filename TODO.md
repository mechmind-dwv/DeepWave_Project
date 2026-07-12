# 📋 DeepWave — Pendientes (actualizado 10 julio 2026)

## 🎯 Objetivo actual: ampliar dataset real a mínimo 350 positivos

**Estado:** 214/350 eventos reales (61%)

**Lección operativa importante:** cuando una descarga falla por
`truncated file`, el archivo `.hdf5` corrupto queda en
`data/eventos_reales/` y el script lo reutiliza sin volver a
descargar — hay que borrarlo EXPLÍCITAMENTE (`rm -f ruta/al/archivo`)
antes de reintentar, o el reintento fallará de nuevo con el mismo
error aunque la red ya esté bien.
**Meta de largo plazo:** miles de eventos (nivel de robustez estadística real)
**Método:** muestreo aleatorio reproducible (semilla fija) desde
`data/candidatos_nuevos_catalogos.json` (245+ candidatos disponibles
de GWTC-4.0/4.1/5.0/O4_Discovery_Papers), en bloques de 15 vía
`construir_dataset_real.py` (con guardado incremental cada 5 eventos
— corregido el 10 julio tras detectar pérdida de progreso en tandas
grandes sin guardado parcial).

### Progreso de la curva de aprendizaje (AUC condicionado por SNR)

| n eventos | SNR alto | SNR bajo | Nota |
|---|---|---|---|
| 40 | 0.754 (agregado) | — | referencia inicial, sin condicionar |
| 75 | 0.728 (agregado) | — | IC95%: 0.653–0.797 |
| 107 | 0.757 | 0.590 | primera vez condicionado por SNR |
| 116 | 0.765 | 0.602 | confirmación 1 |
| 133 | 0.788 | 0.602 | confirmación 2 (SNR bajo estable) |

**Hallazgo consolidado:** el rendimiento depende fuertemente de la
composición del dataset por SNR (p=0.0288, Mann-Whitney U). El AUC
agregado sin condicionar es engañoso. Reportar siempre condicionado
por SNR (alto/bajo respecto a la mediana).

### Próximos hitos de la curva (según vayamos ampliando)
- [ ] n≈150: siguiente checkpoint natural
- [ ] n≈200: repetir AUC condicionado + snapshot versionado
- [ ] n≈250
- [ ] n≈300
- [ ] n≈350: objetivo mínimo de esta fase — evaluar si el patrón
      SNR-alto/SNR-bajo sigue estable o si aparece algo nuevo con
      una muestra de este tamaño
- [ ] Considerar recalcular la correlación H1-L1 (v6) también en cada
      checkpoint grande, para ver si la conclusión de "v1≈v6" se
      mantiene con mucho más datos

## 🟡 Prioridad media (heredado, aún vigente)

- [x] Ampliar validación a más eventos — **en progreso activo**, ver
      objetivo de 350 arriba
- [x] Curva ROC/AUC real — completado, ahora condicionado por SNR
- [x] Entrenar CNN real en Colab — completado, resultado negativo
      (no supera al K-NN con dataset pequeño; revisar si vale la pena
      reintentar una vez el dataset llegue a cientos/miles)

## 🔬 Cadena de experimentos de features (7-8 julio 2026) — cerrada

- [x] v1 (3 features originales): línea base
- [x] v2 (8 sin seleccionar): peor que v1
- [x] v3 (4 seleccionadas por correlación): sigue sin superar v1
- [x] v4/v5 (+ correlación H1-L1): mejora marginal, no concluyente
- [x] v6 (2 selectas + corr. H1-L1): mejor resultado con n=37, pero
      **no se sostuvo** al ampliar a n=68 (v1≈v6, diferencia dentro
      del margen de ruido) — ver sección de veredicto en
      DIVULGACION_PERSEO.md
- [x] **Conclusión:** con este tamaño de dataset, ninguna variante de
      features supera de forma robusta a v1. El camino real de mejora
      es el tamaño del dataset y el condicionamiento por SNR, no más
      ingeniería de features.

## 🏆 Infraestructura consolidada

- [x] `construir_dataset_real.py`: descarga, whitening real, detección
      automática de NaN, **guardado incremental cada 5 eventos** (fix
      10 julio)
- [x] `versionar_dataset.py`: snapshots reproducibles en
      `data/versions/dataset_v_n*/` con metadata (commit git, fecha)
- [x] `auc_condicionado_por_snr.py`: métrica de referencia correcta
      del proyecto
- [x] `bootstrap_auc_v1.py`: intervalos de confianza (2000 remuestreos)
- [x] Dashboard conectado al K-NN real (no modo demo)
- [x] `DeepWaveKNNReferencia`: clasificador dual honesto (H1+L1 / solo H1)

## 🟢 Prioridad baja / limpieza (pendiente, sin urgencia)

- [ ] Mejoras pendientes en construir_dataset_l1.py (retomar cuando se
      recalcule correlación H1-L1 sobre el dataset ampliado a 350):
      guardado incremental cada 10 eventos, informe JSON de eventos
      omitidos con causa (NaN/detector ausente/error red), barra de
      progreso. Ya aplicado: manejo explícito de detector ausente.

- [ ] Migrar dataset_real_*.npy a Git LFS/DVC (sugerencia CodeRabbit) —
      no urgente, evaluar tras alcanzar n=350 y cerrar esta campaña.

- [ ] Revisar commit duplicado `0d4d7d0`/`3407589` en `desarrollo`
- [ ] Confirmar si `deepwave_datos_reales.py` es redundante con
      `deepwave_whitening_real.py`
- [ ] Revisar expiración del token en `~/.git-credentials`
- [ ] Decidir cuándo fusionar `feature/mlp-classifier` a `desarrollo`
      (probablemente al llegar a un hito grande: n=350, o antes si
      se decide congelar el estado actual como release)

## ✅ Hitos completados (resumen histórico)

- [x] Divulgación Perseo↔DeepWave, fusionada a `main`
- [x] Whitening real (Welch + Butterworth), validado contra GW150914
- [x] K-NN nativo sin TensorFlow, funcional en Termux
- [x] MLP descartado (falló control negativo real)
- [x] Dataset real ampliado en 5 tandas: 11→25→40→75→107→116→133 eventos
- [x] Descubrimiento y confirmación (3 puntos) de la estructura AUC
      condicionada por SNR
- [x] IC bootstrap implementado
- [x] Revisión metodológica externa incorporada (matices sobre
      alcance de conclusiones, lenguaje técnico en README, autoría
      transparente)
