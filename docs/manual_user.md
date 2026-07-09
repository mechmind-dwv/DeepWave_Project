# 🌌 DeepWave — Manual de Usuario

Sistema de clasificación de señales de ondas gravitacionales (BBH vs.
Glitch), desarrollado y validado íntegramente en Termux/Android.

## 🚀 Inicio Rápido

```bash
git clone https://github.com/mechmind-dwv/DeepWave_Project.git
cd DeepWave_Project
pip install -r requirements.txt --break-system-packages
python dashboard.py
```
Luego abre: `http://localhost:5000`

## 🏗️ Arquitectura Real (verificada)

| Módulo | Estado | Descripción |
|---|---|---|
| `deepwave_preprocessing.py` | ✅ Funcional | STFT: señal (2048 pts) → espectrograma (103×19) |
| `deepwave_core.py` | ✅ Funcional (juguete) | K-NN con 3 features hardcodeadas, ilustrativo |
| `deepwave_knn_real.py` | ✅ Funcional y validado | K-NN con features reales del espectrograma |
| `deepwave_classifier_cnn_real.py` | ⚠️ Arquitectura definida, **sin entrenar** | Requiere TensorFlow (no disponible en Termux) |
| `deepwave_whitening_real.py` | ✅ Funcional y validado | Whitening real (PSD Welch + Butterworth) sobre datos GWOSC |
| `dashboard.py` | ⚠️ Funcional en **modo demostración** | Import roto a módulos inexistentes; usa resultados aleatorios, no el clasificador real |

## 🧪 Validación Científica Real (no simulada)

Se validó `deepwave_knn_real.py` contra datos públicos reales de LIGO
(evento GW150914), tras aplicar whitening real:

| Segmento | Naturaleza | Predicción | Confianza |
|---|---|---|---|
| GW150914 (evento) | Real (LIGO) | FUSIÓN BBH | 100% |
| Ruido -1000s | Real (LIGO) | GLITCH | 100% |
| Ruido +500s | Real (LIGO) | GLITCH | 100% |

**Límites honestos:** N=1 evento positivo, K=5 vecinos, sin curva
ROC/AUC. Es una prueba de concepto con estructura de control
apropiada, no una validación estadística robusta ni un resultado
listo para publicación científica. Ver `DIVULGACION_PERSEO.md` para
la metodología completa.

## ⚠️ Pendientes conocidos

1. `dashboard.py` no está conectado al clasificador real — corre en
   modo demo con resultados aleatorios.
2. No existe ningún modelo CNN entrenado (`models/` está vacío).
   Entrenar `train_cnn.py` requiere un entorno con TensorFlow
   (Google Colab o PC), no funciona en Termux.
3. Solo se ha validado contra 1 evento real (GW150914). Ampliar a
   más eventos (GW170814, GW190521) daría una validación más sólida.

## 📁 Estructura del Proyecto

```
DeepWave_Project/
├── codigo_fuente/     # Módulos de procesamiento y clasificación
├── data/              # Datos reales descargados + procesados (.npy)
├── docs/              # Este manual
├── templates/         # Interfaz web (dashboard.html)
├── dashboard.py        # Panel interactivo (Flask + Plotly)
└── requirements.txt
```

---
*Este manual refleja el estado real y verificado del proyecto al
5 de julio de 2026. Reemplaza una versión anterior que incluía
métricas de precisión no verificadas.*
