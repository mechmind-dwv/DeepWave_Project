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

## Curva de aprendizaje real: la meseta del K-NN con 3 features

Se amplió el dataset real en tres etapas (11 → 25 → 40 eventos
confirmados de LIGO/Virgo), repitiendo la validación leave-one-out
en cada una:

| Tamaño dataset | Mejor configuración | Recall+ | Recall- | Global |
|---|---|---|---|---|
| 33 muestras (11 eventos) | K=3 sin balancear | 54.5% | 95.5% | 81.8% |
| 75 muestras (25 eventos) | K=3 balanceado | 68.0% | 72.0% | 70.0% |
| 120 muestras (40 eventos) | K=1 balanceado | 67.5% | 72.5% | 70.0% |

**Conclusión honesta:** el salto de 11→25 eventos trajo una mejora
real en Recall+ (54.5%→68%). Pero de 25→40 eventos, el Recall+ se
estancó (68%→67.5%, dentro del margen de ruido). Esto indica que el
cuello de botella ya no es la cantidad de datos, sino el diseño de
las 3 features (energía en banda baja, pendiente temporal, pico
máximo) — probablemente no capturan suficiente información para
distinguir estos casos límite, sin importar cuántos eventos más se
añadan. Duplicar el dataset una vez más (a ~80 eventos) podría no
mejorar significativamente el resultado; el siguiente paso lógico
sería enriquecer las features (por ejemplo, más estadísticos del
espectrograma, o coeficientes de la transformada wavelet) antes de
seguir solo agregando eventos.

**Nota metodológica:** con un solo detector (H1) y ventanas de 32s,
el dataset seguirá siendo pequeño en términos absolutos comparado
con lo que requeriría un entrenamiento profundo real; esta limitación
de escala es inherente al recurso de cómputo (Termux/Android) y al
tiempo de descarga, no un descuido.

## Curva ROC/AUC real (K-NN, 40 eventos, leave-one-out)

Se calculó la curva ROC completa usando un score continuo (proporción
de votos BBH entre los 15 vecinos más cercanos, K=15), en vez del
voto binario mayoritario usado hasta ahora. Validación leave-one-out
sobre las 120 muestras reales (40 positivos + 80 negativos).

**AUC = 0.754**

Para contexto: 0.5 equivale a azar puro, 1.0 a un clasificador
perfecto. Un AUC de 0.754 se considera "aceptable/bueno" en la
literatura estándar de clasificación binaria — un resultado honesto
y creíble para un K-NN con solo 3 features artesanales sobre un
dataset de 40 eventos reales.

**Punto de operación destacado:** con umbral 0.400 (40% de los 15
vecinos votando BBH), se obtiene TPR=60% con FPR=7.5% — un mejor
compromiso que el punto de operación por defecto (K=3, voto
mayoritario simple), que da menor sensibilidad a cambio de mayor
especificidad.

Gráfico disponible en `docs/curva_roc_deepwave.png`.

## CNN entrenada en Colab con dataset real: no supera al K-NN

Se entrenó `deepwave_classifier_cnn_real.py` en Google Colab (con GPU,
ya que TensorFlow no corre en Termux) usando el mismo dataset real de
120 muestras (40 eventos + 80 negativos) que valida el K-NN. Se usó
validación 5-fold estratificada con EarlyStopping.

**Resultado k-fold:** 67.5% ± 1.7% de precisión global — muy cercano
al 66.7% que se obtiene prediciendo siempre "glitch" (la clase
mayoritaria, 2:1). Diagnóstico con matriz de confusión sobre un fold
representativo:

| | Predicho: Glitch | Predicho: BBH |
|---|---|---|
| **Real: Glitch** (16) | 14 | 2 |
| **Real: BBH** (8) | 6 | 2 |

Recall+ (detección real de eventos) = 25%, muy por debajo del 67.5%
que logra el K-NN con las mismas 120 muestras.

**Conclusión honesta:** con solo 96-120 muestras de entrenamiento, la
CNN no tiene datos suficientes para que las capas convolucionales
aprendan patrones útiles — el K-NN con 3 features artesanales
generaliza mejor a esta escala. Esto es coherente con la literatura:
las redes convolucionales típicamente requieren miles de ejemplos
para superar métodos más simples. El modelo final (entrenado con
todos los datos, sin held-out) llegó a 95%+ de "precisión" — clásico
sobreajuste severo, no una métrica válida.

**Decisión:** el K-NN (`deepwave_knn_real.py`) sigue siendo el
clasificador recomendado para este proyecto en su estado actual. La
CNN queda documentada como experimento con resultado negativo, útil
para saber que el camino de "más complejidad de modelo" no ayuda
mientras el dataset siga siendo pequeño — el camino real de mejora es
ampliar el dataset (más eventos) o mejorar las features, no usar un
modelo más grande.

## Features enriquecidas (v2, 8 estadísticos): resultado honesto negativo

Se probó ampliar las features de 3 a 8 (añadiendo energía por banda
media/alta, varianza, entropía espectral, número de picos), sobre el
mismo dataset real de 40 eventos, con la hipótesis de que el diseño
de features —no el tamaño del dataset— era el cuello de botella.

| Configuración | v1 (3 feat.) Recall+ | v2 (8 feat.) Recall+ |
|---|---|---|
| K=1, sin balancear | 40.0% | 42.5% |
| K=3, sin balancear | 52.5% | 37.5% |
| K=1, balanceado | **67.5%** | 57.5% |
| K=3, balanceado | 65.0% | 57.5% |
| K=5, balanceado | 57.5% | 47.5% |

**Resultado honesto: v2 NO supera a v1.** En el mejor punto de
operación de cada versión, v1 (67.5% Recall+) supera claramente a v2
(57.5%). Esto contradice la hipótesis previa de que "más features
romperían la meseta" — sugiere en cambio que algunas de las 5
features nuevas (probablemente entropía espectral, que mostró muy
poca variación entre clases incluso en datos sintéticos) añaden ruido
dimensional que diluye la señal útil de las 3 features originales en
el cálculo de distancia K-NN, en vez de aportar información nueva.

**Conclusión honesta revisada:** el cuello de botella no es
simplemente "pocas features" ni "pocos datos" por separado — es más
probable que sea una combinación de dataset pequeño (imposibilita
validar qué features realmente generalizan) y features que no fueron
seleccionadas con un criterio estadístico riguroso (se diseñaron por
intuición, no por selección de features basada en el propio
dataset). El camino más honesto hacia adelante sería un análisis de
importancia de features (ej. correlación individual de cada feature
con la etiqueta) antes de añadir más, en vez de agregar más
estadísticos a ciegas.

## Selección de features por correlación (v3): tampoco rompe la meseta

Se midió la correlación individual de cada una de las 8 features (v2)
con la etiqueta real. Resultado: `pico_max` (+0.569), `energia_media`
(+0.348), `energia_baja` (+0.308) mostraron correlación notable;
`entropia_espectral`, `varianza_total`, `pendiente`, `num_picos`
resultaron prácticamente ruido (|corr| < 0.11) — incluyendo
`pendiente`, una de las 3 features originales de v1.

Se construyó v3 con solo las 3-4 features útiles y se repitió el
leave-one-out:

| Versión | Mejor Recall+ | Recall- en ese punto |
|---|---|---|
| v1 (3 originales) | **67.5%** | 72.5% |
| v2 (8, sin seleccionar) | 57.5% | 77.5-82.5% |
| v3 (4, seleccionadas por correlación) | 60.0% | 77.5-80.0% |

**Conclusión honesta:** ni v2 ni v3 superan a v1. Esto descarta la
hipótesis de que el diseño de features sea el cuello de botella
principal — la correlación individual no captura interacciones entre
features, y es posible que el límite real sea más fundamental: un
solo detector (H1) puede no llevar suficiente información
discriminante, independientemente de cómo se procese. El siguiente
paso lógico es incorporar el segundo detector (L1) y aprovechar la
coincidencia temporal entre ambos — el método que LIGO usa realmente
para confirmar eventos reales, y que nuestro pipeline actual no
utiliza en absoluto.

## Primer resultado positivo: correlación cruzada H1-L1 (coincidencia entre detectores)

Tras 3 intentos sin éxito (MLP, features v2, features v3), se probó
añadir la correlación cruzada temporal entre los dos detectores LIGO
(H1 Hanford, L1 Livingston) — el principio real que usa LIGO para
confirmar eventos genuinos, ausente hasta ahora en el pipeline.

**Test estadístico (Mann-Whitney U, 37 eventos con H1+L1 disponibles):**
correlación media en eventos reales = 0.148, en ruido = 0.129.
**p-valor = 0.0025** — diferencia estadísticamente significativa.

**Resultado en el K-NN** (mismo subconjunto de 37 eventos, v1=solo H1
vs v4=H1+correlación H1-L1):

| Config | v1 Recall+ | v4 Recall+ |
|---|---|---|
| K=1, balanceado | 59.5% | **67.6%** |
| K=1, sin balancear | 45.9% | **59.5%** |

**Conclusión honesta:** primera mejora consistente de la sesión.
Aunque la correlación cruzada individual es estadísticamente
significativa pero de magnitud modesta, combinada con las features
espectrales originales mejora el Recall+ en 8-14 puntos porcentuales
en los mejores puntos de operación. Esto confirma la hipótesis:
un solo detector no lleva suficiente información — la coincidencia
entre detectores, el método real de LIGO, sí aporta señal genuina.

**Limitación:** 3 de 40 eventos no tienen H1 disponible (solo
L1+Virgo ese día), así que esta comparación es sobre un subconjunto
de 37, no los 40 completos.

## Refinamiento de la correlación H1-L1: mejor p-valor no implica mejor clasificador

Se probó restringir la búsqueda de correlación cruzada a la ventana
físicamente plausible (±10ms, tiempo de viaje de la luz entre
Hanford y Livingston), en vez de buscar en toda la ventana de 1s.

**Resultado del test estadístico:** mejora dramática (p=0.0025 →
p≈0.0000; separación de medias 0.148/0.129 → 0.128/0.085).

**Resultado en el K-NN (Recall+, mismo subconjunto de 37 eventos):**

| Config | v1 (solo H1) | v4 (corr. amplia) | v5 (corr. restringida) |
|---|---|---|---|
| K=1, balanceado | 59.5% | **67.6%** | 62.2% |
| K=3, sin balancear | 43.2% | 40.5% | 48.6% |

**Conclusión honesta:** v5, pese a tener un p-valor muchísimo mejor,
**no supera a v4** en el punto de operación principal. La versión
restringida separa mejor las medias poblacionales, pero tiene mayor
dispersión relativa (std=0.049 vs 0.038), lo que puede confundir al
K-NN en decisiones caso-por-caso. Lección metodológica: un test de
significancia (p-valor) mide si una diferencia es real, no si esa
feature es más útil para un clasificador — son preguntas distintas.
Se conserva v4 (correlación de ventana amplia) como la versión en
uso, dado su mejor desempeño práctico medido con leave-one-out.

## v6: mejor resultado de la sesión (combinando hallazgos previos)

Se combinaron los dos hallazgos positivos de la cadena de
experimentos: las 2 features individuales con mayor correlación
(`pico_max`=+0.569, `energia_media`=+0.348) más la correlación
cruzada H1-L1 de ventana amplia (v4).

**Resultado (K=1 balanceado, subconjunto de 37 eventos con H1+L1):**
Recall+ = **70.3%**, Global = 73.0%, Recall- = 75.7%.

| Versión | # Features | Mejor Recall+ |
|---|---|---|
| v1 (original) | 3 | 67.5% |
| v2 (sin seleccionar) | 8 | 57.5% |
| v3 (seleccionadas por correlación) | 4 | 60.0% |
| v4 (v1 + corr. H1-L1 amplia) | 4 | 67.6% |
| v5 (v1 + corr. H1-L1 restringida) | 4 | 62.2% |
| **v6 (2 selectas + corr. H1-L1)** | **3** | **70.3%** |

**Conclusión de la cadena completa:** ni "más features" (v2) ni
"features seleccionadas por correlación individual sin combinar con
detección multi-detector" (v3) mejoraron el resultado. La mejora real
vino de combinar **menos pero mejores features individuales** con
**información genuinamente nueva** (coincidencia entre detectores),
no de acumular estadísticos del mismo espectrograma de un solo
detector. Es el resultado más sólido de la sesión y el candidato a
convertirse en el clasificador de referencia del proyecto.

## Validación k-fold y ROC/AUC de v6: confirma la mejora, expone la inestabilidad

**AUC:** v6 = 0.758, v1 = 0.754 — mejora marginal pero consistente con
el Recall+ superior visto en leave-one-out (confirmación cruzada con
método independiente).

**K-fold estratificado (5 folds, K=1):** Recall+ media=60.7%,
**std=28.4%** — rango de 14.3% a 87.5% entre folds. Esta inestabilidad
extrema es consecuencia directa del tamaño de muestra: cada fold de
prueba contiene solo 7-8 positivos, así que 1-2 casos difíciles
pueden dominar el resultado de un fold completo.

**Conclusión honesta:** el AUC (que promedia sobre todos los umbrales
y usa leave-one-out, con 111 evaluaciones individuales) es la métrica
más estable y confiable de las que tenemos. El k-fold de 5 particiones,
aunque metodológicamente válido, es demasiado grueso para un dataset
de este tamaño — sus resultados por fold individual no deben
sobre-interpretarse. **La limitación de fondo de todo el proyecto
sigue siendo el tamaño del dataset (37-40 eventos)**, no el método de
clasificación ni las features elegidas. Cualquier futura ampliación
del dataset (GWTC-3 completo, ~35 eventos más) sería la mejora que
más reduciría esta inestabilidad, más que seguir refinando features.

## Dataset duplicado (37→68 eventos con H1+L1): la estabilidad revela la verdad

Al casi duplicar el dataset (GWTC-3 añadido), se repitió la
validación de v6:

| Métrica | 37 eventos | 68 eventos |
|---|---|---|
| AUC | 0.758 | 0.708 |
| Recall+ k-fold (media) | 60.7% | 54.3% |
| Recall+ k-fold (std) | 28.4% | **8.9%** |

**Conclusión honesta y más importante de la sesión:** el AUC de 0.758
y el Recall+ de 70.3% obtenidos con 37 eventos estaban probablemente
**inflados por varianza de muestra pequeña** — la std de 28.4% ya lo
advertía. Con 68 eventos, la estimación es más baja pero mucho más
confiable (std=8.9%, tres veces menor). Esta es la razón por la que
la validación rigurosa (k-fold, no solo un leave-one-out puntual) y
la ampliación del dataset son más valiosas que seguir afinando
features: un resultado "mejor" en una muestra pequeña puede
simplemente ser ruido que desaparece con más datos.

**Estado real del proyecto:** el clasificador v6 (H1+L1) da Recall+
≈54-67% dependiendo de la muestra y el método de validación — un
rango honesto, no un número único inflado. v1 (solo H1) sigue siendo
competitivo (AUC=0.754 vs 0.708 de v6 en el dataset ampliado), lo
que sugiere que la correlación H1-L1 aporta menos de lo que el
resultado inicial con 37 eventos sugería.

## Veredicto final: v1 y v6 convergen, sin ganador claro (75 eventos)

Comparación metodológicamente justa (mismo protocolo k-fold + AUC,
máximo de datos disponible para cada versión):

| | v1 (75 eventos, solo H1) | v6 (68 eventos, H1+L1) |
|---|---|---|
| AUC | 0.728 | 0.708 |
| Recall+ k-fold (media ± std) | 50.7% ± 9.0% | 54.3% ± 8.9% |
| Global k-fold (media) | 68.0% | 65.6% |

**Conclusión honesta y definitiva de la cadena de experimentos:**
v1 y v6 son estadísticamente indistinguibles con este tamaño de
dataset — la diferencia observada está dentro del margen de ruido
(std≈9% en ambos). No hay evidencia suficiente para afirmar que una
arquitectura supere a la otra únicamente con estos resultados.

**Matiz importante sobre el alcance de esta conclusión:** que la
correlación H1-L1 no mejore *esta implementación concreta* (una
única feature de correlación cruzada temporal, alimentando un K-NN
con features espectrales simples) NO implica que la información
conjunta de múltiples detectores carezca de utilidad en general. En
los pipelines profesionales de LIGO/Virgo, la coincidencia entre
detectores es central para la confirmación de eventos reales, pero
se explota con técnicas mucho más sofisticadas — coincidencia
temporal precisa, matched filtering, estimación bayesiana, análisis
coherente multi-detector — no añadiendo una sola característica
resumen a un clasificador simple. La conclusión aquí es acotada:
"con este método y este dato no hubo mejora reproducible", no "la
información H1-L1 no sirve".

**También se revisa la cifra de referencia previa:** el descenso de
AUC≈0.754 (40 eventos) a AUC≈0.728 (75 eventos) es coherente con un
efecto esperado de estimación en muestras pequeñas — no implica que
el modelo haya empeorado, sino que la estimación anterior era menos
robusta y más optimista. El resultado más importante metodológicamente
no es el valor puntual de AUC, sino que el comportamiento del modelo
se ha estabilizado al aumentar el tamaño de muestra (std del k-fold
bajó de 28.4% a 8.9-9.0% en ambas versiones) — esa estabilización es
la señal real de que ahora tenemos una estimación confiable.

**Recomendación de ingeniería:** usar v1 (solo H1) como clasificador
de referencia por simplicidad — evita la complejidad de manejar dos
detectores y el modo dual, sin sacrificar rendimiento real. v6 queda
documentado como un experimento válido que no demostró ventaja
suficiente para justificar su complejidad adicional.

**El techo real del proyecto** con el método actual (K-NN, 3-4
features espectrales simples, ~75-150 eventos reales) parece estar
en **AUC≈0.72-0.75, Recall+≈50-55%** — un resultado honesto, muy por
encima del azar (AUC=0.5), pero lejos de un sistema listo para
detección profesional (que requeriría matched filtering, bancos de
templates, y miles de eventos de entrenamiento).

**Intervalo de confianza bootstrap (2000 remuestreos, v1, n=75+150):**

**AUC = 0.728 (IC 95%: 0.653–0.797)**

El límite inferior del IC (0.653) confirma que el modelo aporta señal
real, claramente por encima del azar (0.5), incluso en el escenario
más conservador. El límite superior (0.797) recuerda que la
incertidumbre restante es considerable — no se puede descartar que el
verdadero rendimiento sea sustancialmente más modesto que el AUC
puntual sugiere. Este intervalo, no el número puntual aislado, es el
reporte metodológicamente correcto del rendimiento del baseline.

**Intervalo de confianza bootstrap (2000 remuestreos, v1, n=75+150):**

**AUC = 0.728 (IC 95%: 0.653–0.797)**

El límite inferior del IC (0.653) confirma que el modelo aporta señal
real, claramente por encima del azar (0.5), incluso en el escenario
más conservador. El límite superior (0.797) recuerda que la
incertidumbre restante es considerable — no se puede descartar que el
verdadero rendimiento sea sustancialmente más modesto que el AUC
puntual sugiere. Este intervalo, no el número puntual aislado, es el
reporte metodológicamente correcto del rendimiento del baseline.

## Curva de aprendizaje real: el AUC BAJA de forma consistente, no se estabiliza

| Positivos | AUC | IC95% |
|---|---|---|
| 40 | 0.754 | — |
| 75 | 0.728 | 0.653–0.797 |
| 107 | 0.687 | 0.623–0.747 |

**Hallazgo honesto e importante:** el AUC desciende de forma
consistente en 3 puntos consecutivos, no se estabiliza. Esto es
distinto de lo esperado si el descenso previo (754→728) fuera solo
efecto de estimación en muestra pequeña — de ser así, el AUC debería
estabilizarse, no seguir cayendo.

**Hipótesis a investigar:** los eventos añadidos en la última tanda
provienen de catálogos más recientes (GWTC-4.0/4.1/5.0, campaña de
observación O4), que podrían incluir sistemáticamente eventos de
menor SNR, mayor distancia, o rangos de masa distintos a los del
catálogo original GWTC-1 (con el que se diseñaron y calibraron las
features). Antes de seguir ampliando el dataset, sería más informativo
verificar si los eventos añadidos recientemente tienen SNR
sistemáticamente más bajo que los originales — separando el efecto
"más datos" del efecto "datos más difíciles".
