# 🌌 DEEPWAVE: Sistema de Detección de Fusión de Binarias de Agujeros Negros (BBH) Mediante IA

## 🔭 Visión General del Proyecto

DeepWave representa un salto cuántico en el análisis de datos de **Ondas Gravitacionales (GW)** provenientes de observatorios como LIGO/Virgo. El objetivo es superar el desafío del ruido (*glitches*) mediante el uso de una **Red Neuronal Convolucional (CNN)** diseñada a medida, entrenada para reconocer la sutil firma del patrón de "chirp" de una Fusión de Binarias de Agujeros Negros (BBH) en su representación espectral.

Este sistema reduce la tasa de falsos positivos y acelera la confirmación de eventos GW, permitiendo a la humanidad observar el universo en sus momentos más violentos y fundamentales.

## 🤝 Liderazgo y Colaboración Científica

| Rol | Autor / Entidad | Contribución Principal |
| :--- | :--- | :--- |
| **Investigador Principal** | **Dr. Benjamin Cabeza Duran** | Definición del Problema GW y Validación de Datos |
| **Arquitecto de IA / Cognición** | **Gemini IA - Unidad de Análisis Cuántico** | Diseño de la Arquitectura CNN, Modelo K-NN y Lógica STFT |
| **Metodología de Señales** | **GWOSC (Simulado)** | Generación y Normalización de Series Temporales (Ruido y BBH) |

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

*(C) Octubre 2025. DeepWave Project. Gemini IA - Unidad de Análisis Cuántico.*
