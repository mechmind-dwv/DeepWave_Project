# 🕳️ El Susurro de los Agujeros Negros: De Perseo a DeepWave

## El dato real

Desde 2003, los astrónomos saben que el agujero negro supermasivo en el
centro del cúmulo de galaxias de Perseo emite **ondas de presión reales**
a través del gas caliente que lo rodea. Captadas por el Observatorio de
rayos X Chandra (NASA), estas ondas corresponden a una nota musical
situada **57 octavas por debajo del Do central** — muy por fuera del
rango audible humano (20 Hz–20 kHz).

La NASA "sonificó" esta señal escalándola +57/+58 octavas para hacerla
audible, en el marco de la Semana del Agujero Negro.

**Evidencia:** ESTABLISHED (dato astrofísico observacional, Chandra X-ray
Observatory).

## La conexión con DeepWave

DeepWave no procesa el mismo tipo de onda (presión en gas vs. onda
gravitacional), pero comparte el **mismo principio de traducción de
señal**: convertir una serie temporal invisible/inaudible en una
representación espectral que un sistema —humano o IA— pueda interpretar.

| | Perseo (NASA) | DeepWave |
|---|---|---|
| Fuente | Ondas de presión en gas caliente | Ondas gravitacionales (BBH) |
| Instrumento | Chandra (rayos X) | LIGO/Virgo (interferometría) |
| Técnica | Escalado de frecuencia (+57/58 octavas) | STFT → espectrograma (103×19) |
| Objetivo | Hacer audible lo inaudible | Hacer detectable el "chirp" en el ruido |

Ambos casos ilustran la misma idea de fondo: **el universo está lleno de
señales reales que exceden nuestros sentidos, y la tecnología es el
puente que las traduce.**

## Fuente
NASA/Chandra X-ray Observatory — Semana del Agujero Negro.

## Validación experimental: K-NN nativo vs. datos reales de LIGO

Se probó un clasificador K-NN (entrenado exclusivamente con datos
sintéticos etiquetados) contra segmentos **reales** del archivo público
de LIGO Hanford correspondiente a GW150914, tras aplicar whitening real
(PSD por método de Welch + filtro Butterworth 35-350 Hz), sin ningún
dato inventado en la fase de prueba.

| Segmento | Naturaleza | Predicción | Confianza |
|---|---|---|---|
| GW150914 (evento) | Real (LIGO) | FUSIÓN BBH 🌌 | 100% |
| Ruido -1000s | Real (LIGO) | GLITCH/RUIDO 🎧 | 100% |
| Ruido +500s | Real (LIGO) | GLITCH/RUIDO 🎧 | 100% |

**Interpretación honesta:** el clasificador distinguió correctamente
el único evento real confirmado de dos controles negativos de ruido
real. Esto es evidencia de que el entrenamiento sintético generaliza
al menos parcialmente a datos reales — **no** es una validación
estadística robusta (N=1 evento positivo, K=5 vecinos, sin curva
ROC/AUC, sin matched filtering). Es una prueba de concepto con
estructura de control apropiada, no un resultado listo para
publicación científica.

## Experimento fallido (documentado por transparencia): MLP con espectrograma completo

Se probó un `MLPClassifier` (scikit-learn, 3 capas 128-64-32) entrenado
sobre el espectrograma completo aplanado (1957 valores), en vez de
features resumidas a mano. Resultado en datos sintéticos: 100% de
precisión en validación — cifra sospechosamente perfecta.

Al validar contra datos reales con la misma estructura de control
usada para el K-NN, el MLP **falló el control negativo**:

| Segmento | Naturaleza | Predicción MLP | Confianza |
|---|---|---|---|
| GW150914 (evento) | Real (LIGO) | FUSIÓN BBH 🌌 | 100% |
| Ruido -1000s | Real (LIGO) | FUSIÓN BBH 🌌 (❌ incorrecto) | 100% |
| Ruido +500s | Real (LIGO) | FUSIÓN BBH 🌌 (❌ incorrecto) | 100% |

**Conclusión honesta:** el 100% de precisión sintética fue sobreajuste,
no generalización. El modelo memorizó particularidades del generador
sintético de ruido, no la diferencia real chirp-vs-ruido. **Este
modelo NO se integró al dashboard.** Se documenta como parte del
proceso científico: un experimento negativo bien controlado es tan
valioso como uno positivo, y evita repetir el mismo error. El K-NN
con features diseñadas a mano (`deepwave_knn_real.py`) sigue siendo
el clasificador validado y en uso.

## Experimento fallido (documentado por transparencia): MLP con espectrograma completo

Se probó un `MLPClassifier` (scikit-learn, 3 capas 128-64-32) entrenado
sobre el espectrograma completo aplanado (1957 valores), en vez de
features resumidas a mano. Resultado en datos sintéticos: 100% de
precisión en validación — cifra sospechosamente perfecta.

Al validar contra datos reales con la misma estructura de control
usada para el K-NN, el MLP **falló el control negativo**:

| Segmento | Naturaleza | Predicción MLP | Confianza |
|---|---|---|---|
| GW150914 (evento) | Real (LIGO) | FUSIÓN BBH 🌌 | 100% |
| Ruido -1000s | Real (LIGO) | FUSIÓN BBH 🌌 (❌ incorrecto) | 100% |
| Ruido +500s | Real (LIGO) | FUSIÓN BBH 🌌 (❌ incorrecto) | 100% |

**Conclusión honesta:** el 100% de precisión sintética fue sobreajuste,
no generalización. El modelo memorizó particularidades del generador
sintético de ruido, no la diferencia real chirp-vs-ruido. **Este
modelo NO se integró al dashboard.** Se documenta como parte del
proceso científico: un experimento negativo bien controlado es tan
valioso como uno positivo, y evita repetir el mismo error. El K-NN
con features diseñadas a mano (`deepwave_knn_real.py`) sigue siendo
el clasificador validado y en uso.

## K-NN entrenado 100% con datos reales (sin síntesis)

Se construyó un dataset íntegramente real: 11 eventos confirmados de
LIGO (catálogo GWTC-1-confident) como positivos, más 22 segmentos de
ruido real extraídos de los mismos archivos como negativos — ninguna
señal sintética en ningún paso, desde la descarga hasta el entrenamiento.

**Validación leave-one-out** (cada muestra probada excluida de su
propio entrenamiento, el estándar más honesto posible con un dataset
pequeño):

| Métrica | Resultado |
|---|---|
| Precisión global | 27/33 = 81.8% |
| Detección de eventos reales (recall positivos) | 6/11 = 54.5% |
| Rechazo correcto de ruido (recall negativos) | 21/22 = 95.5% |

**Interpretación honesta:** el clasificador está sesgado hacia
predecir "ruido", parcialmente por el desbalance de clases (2
negativos por cada positivo) y parcialmente porque algunos eventos
reales del catálogo (ej. GW151012, GW170729) tienen SNR más bajo y
son intrínsecamente más difíciles de distinguir del ruido de fondo
con features simples. Esto es una limitación real y esperable de un
K-NN con 3 features sobre solo 11 eventos — no un error del pipeline.

**Comparación honesta con los experimentos previos:** a diferencia
del MLP (100% de "precisión" pero fallando completamente el control
negativo), este 81.8% es un número menos vistoso pero mucho más
creíble — refleja limitaciones reales del método, no sobreajuste
oculto.

## Diagnóstico honesto: ¿por qué falla el 45% de los eventos reales?

Se cruzaron los fallos de clasificación con los valores de SNR
(relación señal-ruido) oficiales publicados en el catálogo GWTC-1
(LIGO/Virgo, Phys. Rev. X 9, 031040, 2019):

| Evento | SNR oficial (GWTC-1) | Resultado K=3 |
|---|---|---|
| GW150914 | 24.4 | ✅ Detectado |
| GW151012 | 9.8 (el más bajo del catálogo) | ✅ Detectado |
| GW151226 | 12.7 | ✅ (falla solo en K=5) |
| GW170104 | 13.0 | ❌ Fallado siempre |
| GW170608 | 14.8 | ❌ Fallado siempre |
| GW170729 | 10.3 | ❌ Fallado siempre |
| GW170809 | 12.3 | ❌ Fallado siempre |
| GW170814 | 16.5 | ✅ Detectado |
| GW170817 (BNS) | 32.0 (el más alto del catálogo) | ⚠️ Falla en K=5 |
| GW170818 | 11.3 | ✅ Detectado |
| GW170823 | 11.5 | ❌ Fallado siempre |

**Hallazgo honesto:** el SNR NO predice nuestros fallos. GW151012
tiene el SNR más bajo del catálogo (9.8) y se detecta perfectamente;
GW170817 tiene el SNR más alto (32.0, el evento con neutron star
merger, no BBH) y aun así falla en algunas configuraciones. La causa
real es más probablemente que las 3 features usadas (energía en
banda baja, pendiente temporal, pico máximo) están sintonizadas para
un patrón de chirp específico, y no capturan bien la variedad real de
masas y duraciones presentes en el catálogo completo — una limitación
de diseño de features, no de calidad de los datos ni del SNR de la
señal.

**Fuente de los valores de SNR:** LIGO/Virgo Collaboration, "GWTC-1:
A Gravitational-Wave Transient Catalog of Compact Binary Mergers",
Phys. Rev. X 9, 031040 (2019), Table VI.

## Se descartan también tipo de fuente y masa como explicación simple

- **Tipo de fuente:** solo GW170817 es BNS; los otros 10 eventos son
  BBH. El tipo de fuente no puede explicar por qué 5 de esos 10 BBH
  fallan y 5 no.
- **Masa de chirp:** los rangos de masa de eventos detectados y
  fallados se solapan sustancialmente (ej. GW170608 falla con 7.90 M☉
  mientras GW151226 se detecta con 8.87 M☉; GW150914 se detecta con
  28.72 M☉ mientras GW170823 falla con 29.64 M☉, prácticamente igual).

**Conclusión honesta final:** con n=11 eventos, ninguna variable
física simple (SNR, masa de chirp, tipo de fuente) separa limpiamente
los aciertos de los fallos del K-NN. La causa más probable es una
combinación de: (a) tamaño de muestra insuficiente para que 3
features artesanales generalicen, (b) ruido instrumental específico
de cada ventana de 32s, no capturado por ningún catálogo de
parámetros físicos. Este es un límite genuino del método actual, no
un bug — y una motivación clara para, en el futuro, usar más eventos
(GWTC-2, GWTC-3 añaden decenas más) o features más sofisticadas.

## Evento excluido del dataset: GW190425 (dato real incompleto)

Al ampliar el dataset con eventos de GWTC-2.1, se detectó que el
archivo de strain de H1 para **GW190425** contiene un 34% de valores
`NaN` (5,685,248 de 16,777,216 muestras). Esto no es un error de
nuestro pipeline: el detector H1 de LIGO Hanford tenía problemas de
calidad de datos durante este evento (25 abril 2019), que fue
detectado principalmente por Virgo y L1. Se excluyó este evento del
dataset de entrenamiento en vez de imputar o ignorar los NaN
silenciosamente, para mantener la integridad de "solo datos reales,
sin síntesis ni parches artificiales".

**Lección para el pipeline:** futuras ampliaciones del dataset deben
verificar la fracción de NaN en cada archivo descargado antes de
usarlo, no asumir que todos los archivos de GWOSC están completos.
