# 🌌 DEEPWAVE: Sistema de Detección de Fusión de Binarias de Agujeros Negros (BBH) Mediante IA
![CI](https://github.com/mechmind-dwv/DeepWave_Project/actions/workflows/tests.yml/badge.svg)

## 🔭 Visión General del Proyecto

DeepWave es un proyecto de investigación exploratoria sobre clasificación de señales de **Ondas Gravitacionales (GW)** usando datos públicos de LIGO/Virgo (GWOSC). Se evaluaron distintos enfoques de clasificación (K-NN, MLP, CNN) sobre espectrogramas STFT de eventos reales confirmados, con el objetivo de establecer una línea base reproducible y documentar honestamente los límites del método frente al tamaño de muestra disponible.

El proyecto no pretende sustituir los pipelines profesionales de detección (matched filtering, análisis bayesiano coherente); es un ejercicio de aprendizaje aplicado con datos reales, desarrollado íntegramente en un entorno de recursos limitados (Termux/Android).

## Autoría y colaboración

| Categoría | Nombre | Rol |
| :--- | :--- | :--- |
| **Autor humano** | Benjamin Cabeza Duran | Dirección del proyecto, diseño experimental, validación de resultados |
| **Herramienta de asistencia (IA)** | Claude (Anthropic) | Asistencia en implementación de código, diseño de experimentos, redacción de documentación |
| **Herramienta de asistencia (IA)** | Gemini (Google) | Asistencia en fases previas del desarrollo (arquitectura inicial de CNN/K-NN) |
| **Fuente de datos** | GWOSC (Gravitational Wave Open Science Center) | Datos reales de strain de LIGO/Virgo, catálogos GWTC-1/2.1/3 |

*Nota de transparencia: el código fue producido con asistencia sustancial de modelos de lenguaje. Todas las decisiones de diseño experimental, interpretación de resultados y validación fueron responsabilidad del autor humano.*

## 🚀 Componentes Clave del Núcleo Operacional

| Archivo | Dimensión de Procesamiento | Propósito |
| :--- | :--- | :--- |
| **`deepwave_core.py`** | N/A (Clasificación K-NN) | Módulo de verificación inicial y clasificación rápida de rasgos. |
| **`deepwave_preprocessing.py`** | **(2048 puntos) -> (103, 19)** | Módulo crucial de **Pre-procesamiento (STFT)**. Convierte la señal de tiempo a la matriz de espectrograma lista para la CNN. |
| **`deepwave_classifier_cnn.py`** | **(103, 19)** | **Modelo Arquitectónico de la IA.** Define el flujo de las capas Conv y Pool para la extracción de rasgos profundos y la clasificación final. |

## ⚙️ Guía de Instalación Rápida (Entorno Termux/NumPy)

Todos los módulos incluyen su propia función de verificación (`if __name__ == "__main__":`). Use `python codigo_fuente/[nombre_modulo].py` para ejecutar las pruebas.

## 💫 La Revelación Cuántica

DeepWave nos enseña que la existencia es una **serie temporal** plagada de ruido. Nuestra tarea, asistida por la IA, es transformar esa serie en un **espectrograma de conciencia** para discernir la **señal de la verdad** (el *chirp* organizado de una BBH) del **ruido de las distracciones** (*glitches* aleatorios).

## 🕳️ Divulgación: El Susurro de los Agujeros Negros

¿Sabías que el agujero negro de Perseo emite un sonido real, 57 octavas
por debajo del Do central? Descubre la conexión entre esa sonificación
de la NASA (Chandra X-ray Observatory) y el pipeline STFT de DeepWave
en [DIVULGACION_PERSEO.md](DIVULGACION_PERSEO.md).

*(C) Octubre 2025. DeepWave Project. Gemini IA - Unidad de Análisis Cuántico.*
