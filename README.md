# 🌌 DeepWave: Clasificación de Ondas Gravitacionales con Datos Reales



![CI](https://github.com/mechmind-dwv/DeepWave_Project/actions/workflows/tests.yml/badge.svg)



**Estado: campaña de validación completada · 350 eventos reales · desarrollado íntegramente en Termux/Android**

---

## Visión general

DeepWave es un proyecto de investigación aplicada sobre clasificación de señales de **ondas gravitacionales** (fusiones de agujeros negros binarios, BBH) usando datos públicos de LIGO/Virgo (GWOSC). El objetivo no es competir con los pipelines profesionales de detección de LIGO/Virgo —que usan matched filtering, bancos de plantillas y análisis bayesiano coherente sobre infraestructura de cómputo masiva— sino establecer, con el mismo rigor metodológico que exigiría cualquier revisión científica, **cuánto se puede lograr con un método simple (K-NN sobre espectrogramas STFT) y datos 100% reales**, documentando honestamente tanto lo que funciona como lo que no.

Todo el desarrollo, entrenamiento y validación se realizó en un dispositivo Android vía Termux, sin GPU, sin IDE, y con una conexión a internet frecuentemente inestable. Esa restricción de recursos es parte de la historia del proyecto, no una excusa para bajar el estándar de la ciencia.

## Resultados actuales

**Dataset:** 350 eventos reales confirmados de LIGO/Virgo (catálogos GWTC-1, GWTC-2, GWTC-2.1, GWTC-3, GWTC-4.0, GWTC-4.1, GWTC-5.0, O3/O4 Discovery Papers, IAS-O3a) + 700 segmentos de ruido real extraídos de los mismos archivos. Cero datos sintéticos en la fase de validación.

| Métrica | Valor |
|---|---|
| AUC agregado (bootstrap, 2000 remuestreos) | **0.735** (IC 95%: 0.705–0.764) |
| AUC en eventos de SNR alto (n=176) | **0.797** |
| AUC en eventos de SNR bajo (n=172) | **0.662** |
| Precisión leave-one-out global (1050 muestras) | 68.5% |

**El hallazgo metodológico central de este proyecto:** el rendimiento del clasificador depende fuertemente de la relación señal-ruido (SNR) del evento — un patrón físicamente esperado, confirmado de forma reproducible en 8 puntos de muestreo consecutivos (n=107 hasta n=350) a medida que se amplió el dataset. Reportar un único AUC agregado sin condicionar por SNR es engañoso; la práctica correcta, consistente con los estándares de reporte de LIGO/Virgo, es condicionar siempre por la dificultad intrínseca de la señal.

La metodología completa, incluida la curva de aprendizaje punto por punto y la investigación que llevó a este hallazgo, está documentada en [`DIVULGACION_PERSEO.md`](DIVULGACION_PERSEO.md).

## Lo que se probó y no funcionó (documentado, no escondido)

Un proyecto científico honesto reporta sus resultados negativos con el mismo cuidado que los positivos:

- **MLP sobre espectrograma completo:** 100% de precisión en datos sintéticos, pero falló por completo el control negativo con datos reales (clasificaba ruido como señal). Descartado y documentado como ejemplo de sobreajuste.
- **Ingeniería de features (v2, v3):** añadir más estadísticos del espectrograma, o seleccionarlos por correlación individual, no superó a la línea base de 3 features simples.
- **CNN (Keras, entrenada en Google Colab):** con un dataset de 120 eventos, obtuvo Recall+ de solo 25% frente al 67% del K-NN — evidencia clara de que, a esa escala, un modelo más complejo no ayuda. Pendiente reintentar con el dataset actual de 350.
- **Correlación cruzada H1-L1 (coincidencia entre detectores):** mostró una mejora real y estadísticamente significativa (p=0.0025) con muestras pequeñas, pero convergió con la línea base al ampliar el dataset — un ejemplo instructivo de por qué la significancia estadística y la utilidad práctica de un clasificador son preguntas distintas.

## Arquitectura del proyecto

| Módulo | Propósito |
|---|---|
| `construir_dataset_real.py` | Descarga desde GWOSC, whitening real (PSD Welch + filtro Butterworth), detección automática de datos incompletos, guardado incremental |
| `deepwave_preprocessing.py` | STFT: convierte la señal cruda (2048 puntos) en espectrograma (103×19) |
| `deepwave_knn_real.py` | Clasificador K-NN de referencia, con features espectrales validadas empíricamente |
| `versionar_dataset.py` | Snapshots reproducibles del dataset en cada hito (`data/versions/`) |
| `auc_condicionado_por_snr.py` | Métrica de referencia del proyecto: AUC separado por régimen de SNR |
| `bootstrap_auc_v1.py` | Intervalo de confianza (bootstrap) para el AUC reportado |
| `deepwave_classifier_cnn_real.py` / `train_cnn.py` | Arquitectura CNN (Keras); requiere entorno con TensorFlow (Google Colab, no funciona en Termux) |
| `dashboard.py` | Panel interactivo (Flask + Plotly) con demostración en vivo del clasificador |

## Guía rápida

```bash
git clone -b feature/mlp-classifier https://github.com/mechmind-dwv/DeepWave_Project.git
cd DeepWave_Project
pip install -r requirements.txt --break-system-packages
python dashboard.py
```

Cada módulo en `codigo_fuente/` incluye su propia verificación ejecutable (`if __name__ == "__main__":`). Para reproducir la construcción del dataset desde cero, ver `construir_dataset_real.py`; para repetir la validación estadística, ver `auc_condicionado_por_snr.py` y `bootstrap_auc_v1.py`.

## Divulgación

Este proyecto nació de una conexión conceptual con la sonificación del agujero negro de Perseo (NASA/Chandra X-ray Observatory) y con el trabajo del autor en heliobiología y ciclos solares. El hilo narrativo completo, desde la inspiración inicial hasta la metodología científica, está en [`DIVULGACION_PERSEO.md`](DIVULGACION_PERSEO.md) — con una separación clara entre la inspiración (legítima y valiosa como motor de curiosidad) y las afirmaciones científicas (que se sostienen solo con evidencia, no con la inspiración que las originó).

## Autoría

| Categoría | Nombre | Rol |
|---|---|---|
| Autor humano | Benjamín Cabeza Durán | Dirección del proyecto, diseño experimental, validación de resultados, campaña de recolección de datos |
| Asistencia de IA | Claude (Anthropic) | Implementación de código, diseño de experimentos, análisis estadístico, redacción de documentación |
| Asistencia de IA | Gemini (Google) | Arquitectura inicial de CNN/K-NN en fases tempranas del proyecto |
| Fuente de datos | GWOSC (Gravitational Wave Open Science Center) | Datos públicos de strain de LIGO/Virgo |

*Nota de transparencia: el código se produjo con asistencia sustancial de modelos de lenguaje, siguiendo las prácticas actuales de divulgación en proyectos asistidos por IA. Todas las decisiones de diseño experimental, la interpretación de resultados, y la validación crítica de cada hallazgo fueron responsabilidad del autor humano — incluyendo el rechazo de resultados que parecían "demasiado buenos" hasta confirmar su robustez con controles apropiados.*

## Licencia

Ver [`LICENSE`](LICENSE).

---

*Última actualización: 12 de julio de 2026. Este README refleja el estado real y verificado del proyecto tras la campaña de ampliación a 350 eventos reales.*
