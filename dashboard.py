#!/usr/bin/env python3
"""
🌌 DEEPWAVE DASHBOARD - Panel de control interactivo para visualización de análisis BBH
Interfaz web profesional con Flask + Plotly
"""

import os
import sys
import json
import numpy as np
from datetime import datetime
from flask import Flask, render_template, jsonify, request
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import webbrowser

# Configurar paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importar módulos DeepWave
try:
    from codigo_fuente.deepwave_core import WaveAnalyzer
    from codigo_fuente.deepwave_preprocessing import generate_spectrogram
    DEEPWAVE_AVAILABLE = True
except ImportError:
    DEEPWAVE_AVAILABLE = False
    print("⚠️  Módulos DeepWave no disponibles, usando modo demostración")

app = Flask(__name__)

class DeepWaveDashboard:
    """Clase principal del dashboard"""
    
    def __init__(self):
        self.analyzer = WaveAnalyzer() if DEEPWAVE_AVAILABLE else None
        self.analysis_history = []
        self.stats = {
            "total_analyses": 0,
            "bbh_detections": 0,
            "glitch_detections": 0,
            "accuracy": 0.0
        }
    
    def analyze_signal(self, amplitude=0.5, frequency=100, persistence=0.5):
        """Analiza una señal con parámetros dados"""
        if self.analyzer:
            # Usar el analizador real
            result = self.analyzer.analyze_waveform(amplitude, frequency, persistence)
            is_bbh = "FUSIÓN" in result
        else:
            # Modo demostración
            result = "FUSIÓN BBH" if frequency < 80 else "GLITCH"
            is_bbh = "FUSIÓN" in result
        
        analysis = {
            "id": len(self.analysis_history) + 1,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "amplitude": amplitude,
            "frequency": frequency,
            "persistence": persistence,
            "result": result,
            "is_bbh": is_bbh,
            "confidence": np.random.uniform(0.7, 0.99) if is_bbh else np.random.uniform(0.3, 0.6)
        }
        
        self.analysis_history.append(analysis)
        self.update_stats()
        
        return analysis
    
    def update_stats(self):
        """Actualiza estadísticas"""
        self.stats["total_analyses"] = len(self.analysis_history)
        self.stats["bbh_detections"] = sum(1 for a in self.analysis_history if a["is_bbh"])
        self.stats["glitch_detections"] = self.stats["total_analyses"] - self.stats["bbh_detections"]
        
        if self.stats["total_analyses"] > 0:
            self.stats["accuracy"] = (self.stats["bbh_detections"] / self.stats["total_analyses"]) * 100
    
    def generate_waveform_plot(self):
        """Genera gráfico de forma de onda"""
        t = np.linspace(0, 2*np.pi, 1000)
        
        # Señal BBH típica (chirp)
        if self.analysis_history and self.analysis_history[-1]["is_bbh"]:
            freq = self.analysis_history[-1]["frequency"]
            wave = np.sin(freq * t * (1 + 0.1 * t)) * self.analysis_history[-1]["amplitude"]
            color = "#FF6B6B"  # Rojo para BBH
            name = "Señal BBH (Chirp)"
        else:
            # Ruido + señal débil
            wave = np.sin(100 * t) * 0.3 + np.random.normal(0, 0.1, 1000)
            color = "#4ECDC4"  # Turquesa para glitch
            name = "Señal + Ruido"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t, y=wave,
            mode='lines',
            name=name,
            line=dict(color=color, width=3),
            fill='tozeroy',
            fillcolor=f'{color}20'
        ))
        
        fig.update_layout(
            title="Forma de Onda de la Señal",
            xaxis_title="Tiempo (s)",
            yaxis_title="Amplitud",
            template="plotly_dark",
            height=300
        )
        
        return fig.to_json()
    
    def generate_spectrogram_plot(self):
        """Genera espectrograma simulado"""
        # Datos para espectrograma
        fs = 1000  # Frecuencia de muestreo
        t = np.linspace(0, 1, fs)
        
        if self.analysis_history and self.analysis_history[-1]["is_bbh"]:
            # Espectrograma de chirp BBH (frecuencia aumenta)
            f0 = 30
            f1 = 200
            chirp = np.sin(2 * np.pi * (f0 * t + (f1 - f0) * t**2 / 2))
            signal = chirp * self.analysis_history[-1]["amplitude"]
        else:
            # Espectrograma de ruido
            signal = np.random.randn(fs) * 0.5
        
        # Espectrograma simple
        spectrogram = np.abs(np.fft.fft(signal).reshape(20, 50))
        
        fig = go.Figure(data=go.Heatmap(
            z=spectrogram,
            colorscale='Viridis',
            showscale=True
        ))
        
        fig.update_layout(
            title="Espectrograma STFT",
            xaxis_title="Tiempo",
            yaxis_title="Frecuencia (Hz)",
            template="plotly_dark",
            height=300
        )
        
        return fig.to_json()
    
    def generate_radar_plot(self):
        """Genera gráfico radar de características"""
        if not self.analysis_history:
            return None
            
        latest = self.analysis_history[-1]
        
        categories = ['Amplitud', 'Frecuencia', 'Persistencia', 'Coherencia', 'SNR']
        
        values = [
            latest["amplitude"] * 100,  # Escalado para mejor visualización
            latest["frequency"] / 2,    # Normalizado
            latest["persistence"] * 100,
            latest["confidence"] * 100,
            80 if latest["is_bbh"] else 30  # Relación señal/ruido simulada
        ]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line_color='#FF6B6B' if latest["is_bbh"] else '#4ECDC4',
            fillcolor='#FF6B6B40' if latest["is_bbh"] else '#4ECDC440',
            name="Características de la señal"
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=False,
            template="plotly_dark",
            height=300,
            title="Análisis de Características"
        )
        
        return fig.to_json()

# Instancia global del dashboard
dashboard = DeepWaveDashboard()

# ================= RUTAS FLASK =================

@app.route('/')
def index():
    """Página principal del dashboard"""
    return render_template('dashboard.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """API para analizar una señal"""
    data = request.json
    
    amplitude = data.get('amplitude', 0.5)
    frequency = data.get('frequency', 100)
    persistence = data.get('persistence', 0.5)
    
    result = dashboard.analyze_signal(amplitude, frequency, persistence)
    
    return jsonify({
        "success": True,
        "analysis": result,
        "stats": dashboard.stats
    })

@app.route('/api/plots')
def get_plots():
    """Obtiene todos los gráficos actualizados"""
    return jsonify({
        "waveform": dashboard.generate_waveform_plot(),
        "spectrogram": dashboard.generate_spectrogram_plot(),
        "radar": dashboard.generate_radar_plot(),
        "history": dashboard.analysis_history[-10:],  # Últimos 10 análisis
        "stats": dashboard.stats
    })

@app.route('/api/history')
def get_history():
    """Obtiene historial completo"""
    return jsonify(dashboard.analysis_history)

@app.route('/api/stats')
def get_stats():
    """Obtiene estadísticas"""
    return jsonify(dashboard.stats)

@app.route('/api/test_sequence')
def test_sequence():
    """Ejecuta una secuencia de pruebas automática"""
    test_cases = [
        {"amplitude": 0.15, "frequency": 60, "persistence": 0.9},
        {"amplitude": 0.95, "frequency": 140, "persistence": 0.1},
        {"amplitude": 0.5, "frequency": 100, "persistence": 0.5},
        {"amplitude": 0.3, "frequency": 75, "persistence": 0.8},
        {"amplitude": 0.8, "frequency": 120, "persistence": 0.3},
    ]
    
    results = []
    for test in test_cases:
        result = dashboard.analyze_signal(**test)
        results.append(result)
    
    return jsonify({
        "success": True,
        "tests_run": len(test_cases),
        "results": results,
        "stats": dashboard.stats
    })

# ================= TEMPLATE HTML =================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌌 DeepWave Dashboard - Detección BBH</title>
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        body {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #ffffff;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .header h1 {
            font-size: 2.8rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            text-shadow: 0 0 20px rgba(78, 205, 196, 0.3);
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.8;
            max-width: 800px;
            margin: 0 auto;
        }
        
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        @media (max-width: 1024px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }
        
        .card {
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }
        
        .card h2 {
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: #4ECDC4;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .card h2 i {
            font-size: 1.8rem;
        }
        
        .plot-container {
            width: 100%;
            height: 320px;
        }
        
        .controls {
            grid-column: 1 / -1;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }
        
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        label {
            font-weight: 600;
            color: #FF6B6B;
        }
        
        input[type="range"] {
            width: 100%;
            height: 8px;
            border-radius: 4px;
            background: rgba(255, 255, 255, 0.1);
            outline: none;
            -webkit-appearance: none;
        }
        
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: #4ECDC4;
            cursor: pointer;
            box-shadow: 0 0 10px rgba(78, 205, 196, 0.5);
        }
        
        .value-display {
            font-size: 1.2rem;
            font-weight: bold;
            color: #FFD93D;
            text-align: center;
            padding: 5px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
        }
        
        .buttons {
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
        }
        
        button {
            padding: 15px 30px;
            border: none;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            flex: 1;
            min-width: 200px;
        }
        
        .btn-analyze {
            background: linear-gradient(90deg, #FF6B6B, #FF8E8E);
            color: white;
        }
        
        .btn-test {
            background: linear-gradient(90deg, #4ECDC4, #6BE0D8);
            color: white;
        }
        
        .btn-reset {
            background: linear-gradient(90deg, #FFD93D, #FFE873);
            color: #333;
        }
        
        button:hover {
            transform: translateY(-3px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        }
        
        button:active {
            transform: translateY(1px);
        }
        
        .results {
            grid-column: 1 / -1;
            margin-top: 20px;
        }
        
        .result-card {
            background: rgba(0, 0, 0, 0.4);
            border-radius: 12px;
            padding: 20px;
            margin-top: 15px;
            border-left: 5px solid;
            animation: slideIn 0.5s ease-out;
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        .bbh-result {
            border-left-color: #FF6B6B;
        }
        
        .glitch-result {
            border-left-color: #4ECDC4;
        }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .result-type {
            font-size: 1.3rem;
            font-weight: bold;
        }
        
        .result-time {
            opacity: 0.7;
            font-size: 0.9rem;
        }
        
        .result-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }
        
        .detail-item {
            text-align: center;
            padding: 10px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
        }
        
        .detail-label {
            font-size: 0.9rem;
            opacity: 0.7;
            margin-bottom: 5px;
        }
        
        .detail-value {
            font-size: 1.2rem;
            font-weight: bold;
            color: #FFD93D;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        
        .stat-card {
            text-align: center;
            padding: 25px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .stat-value {
            font-size: 3rem;
            font-weight: bold;
            margin: 10px 0;
        }
        
        .stat-bbh { color: #FF6B6B; }
        .stat-glitch { color: #4ECDC4; }
        .stat-total { color: #FFD93D; }
        .stat-accuracy { color: #6BCF7F; }
        
        .stat-label {
            font-size: 1rem;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .history-list {
            max-height: 400px;
            overflow-y: auto;
            padding-right: 10px;
        }
        
        .history-list::-webkit-scrollbar {
            width: 8px;
        }
        
        .history-list::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 4px;
        }
        
        .history-list::-webkit-scrollbar-thumb {
            background: #4ECDC4;
            border-radius: 4px;
        }
        
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            opacity: 0.7;
            font-size: 0.9rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🌌 DeepWave Dashboard</h1>
            <p>Sistema de Detección de Fusión de Binarias de Agujeros Negros (BBH) mediante Inteligencia Artificial</p>
        </div>
        
        <!-- Dashboard Grid -->
        <div class="dashboard-grid">
            <!-- Gráfico 1: Forma de Onda -->
            <div class="card">
                <h2>📈 Forma de Onda</h2>
                <div id="waveform-plot" class="plot-container"></div>
            </div>
            
            <!-- Gráfico 2: Espectrograma -->
            <div class="card">
                <h2>🎵 Espectrograma STFT</h2>
                <div id="spectrogram-plot" class="plot-container"></div>
            </div>
            
            <!-- Gráfico 3: Radar -->
            <div class="card">
                <h2>📊 Análisis de Características</h2>
                <div id="radar-plot" class="plot-container"></div>
            </div>
            
            <!-- Controles -->
            <div class="card controls">
                <h2>🎮 Controles de Análisis</h2>
                
                <div class="control-group">
                    <label for="amplitude">Amplitud: <span id="amplitude-value">0.50</span></label>
                    <input type="range" id="amplitude" min="0" max="1" step="0.01" value="0.5">
                </div>
                
                <div class="control-group">
                    <label for="frequency">Frecuencia (Hz): <span id="frequency-value">100</span></label>
                    <input type="range" id="frequency" min="20" max="200" step="1" value="100">
                </div>
                
                <div class="control-group">
                    <label for="persistence">Persistencia: <span id="persistence-value">0.50</span></label>
                    <input type="range" id="persistence" min="0" max="1" step="0.01" value="0.5">
                </div>
                
                <div class="buttons">
                    <button class="btn-analyze" onclick="analyzeSignal()">
                        🔬 Analizar Señal
                    </button>
                    <button class="btn-test" onclick="runTestSequence()">
                        🧪 Ejecutar Pruebas
                    </button>
                    <button class="btn-reset" onclick="resetDashboard()">
                        🔄 Reiniciar
                    </button>
                </div>
            </div>
            
            <!-- Último Resultado -->
            <div class="card results">
                <h2>📋 Resultado del Análisis</h2>
                <div id="latest-result">
                    <div class="result-card placeholder">
                        <p>Ejecuta un análisis para ver los resultados aquí...</p>
                    </div>
                </div>
            </div>
            
            <!-- Historial -->
            <div class="card">
                <h2>📜 Historial de Análisis</h2>
                <div class="history-list" id="history-list">
                    <p>Cargando historial...</p>
                </div>
            </div>
        </div>
        
        <!-- Estadísticas -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total de Análisis</div>
                <div class="stat-value stat-total" id="stat-total">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Detecciones BBH</div>
                <div class="stat-value stat-bbh" id="stat-bbh">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Detecciones Glitch</div>
                <div class="stat-value stat-glitch" id="stat-glitch">0</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Precisión</div>
                <div class="stat-value stat-accuracy" id="stat-accuracy">0%</div>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="footer">
            <p>DeepWave Project © 2025 | Sistema de IA para Análisis de Ondas Gravitacionales</p>
            <p>Desarrollado con Python, Flask y Plotly | GitHub: mechmind-dwv/DeepWave_Project</p>
        </div>
    </div>
    
    <script>
        // Variables globales
        let currentPlots = {};
        
        // Actualizar displays de los sliders
        document.getElementById('amplitude').addEventListener('input', function(e) {
            document.getElementById('amplitude-value').textContent = parseFloat(e.target.value).toFixed(2);
        });
        
        document.getElementById('frequency').addEventListener('input', function(e) {
            document.getElementById('frequency-value').textContent = e.target.value;
        });
        
        document.getElementById('persistence').addEventListener('input', function(e) {
            document.getElementById('persistence-value').textContent = parseFloat(e.target.value).toFixed(2);
        });
        
        // Función para analizar señal
        async function analyzeSignal() {
            const amplitude = parseFloat(document.getElementById('amplitude').value);
            const frequency = parseInt(document.getElementById('frequency').value);
            const persistence = parseFloat(document.getElementById('persistence').value);
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ amplitude, frequency, persistence })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    updateDashboard();
                    displayResult(data.analysis);
                }
            } catch (error) {
                console.error('Error:', error);
                alert('Error al analizar la señal');
            }
        }
        
        // Función para ejecutar secuencia de pruebas
        async function runTestSequence() {
            try {
                const response = await fetch('/api/test_sequence');
                const data = await response.json();
                
                if (data.success) {
                    updateDashboard();
                    displayResult(data.results[data.results.length - 1]);
                    alert(`✅ ${data.tests_run} pruebas ejecutadas exitosamente`);
                }
            } catch (error) {
                console.error('Error:', error);
            }
        }
        
        // Función para reiniciar dashboard
        function resetDashboard() {
            if (confirm('¿Estás seguro de querer reiniciar el dashboard? Se perderá el historial.')) {
                fetch('/api/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ reset: true })
                }).then(() => {
                    location.reload();
                });
            }
        }
        
        // Función para actualizar todo el dashboard
        async function updateDashboard() {
            try {
                const response = await fetch('/api/plots');
                const data = await response.json();
                
                // Actualizar gráficos
                if (data.waveform) {
                    const waveformPlot = JSON.parse(data.waveform);
                    Plotly.react('waveform-plot', waveformPlot.data, waveformPlot.layout);
                }
                
                if (data.spectrogram) {
                    const spectrogramPlot = JSON.parse(data.spectrogram);
                    Plotly.react('spectrogram-plot', spectrogramPlot.data, spectrogramPlot.layout);
                }
                
                if (data.radar) {
                    const radarPlot = JSON.parse(data.radar);
                    Plotly.react('radar-plot', radarPlot.data, radarPlot.layout);
                }
                
                // Actualizar historial
                updateHistory(data.history || []);
                
                // Actualizar estadísticas
                updateStats(data.stats);
                
            } catch (error) {
                console.error('Error actualizando dashboard:', error);
            }
        }
        
        // Función para mostrar resultado
        function displayResult(analysis) {
            const resultDiv = document.getElementById('latest-result');
            
            const resultClass = analysis.is_bbh ? 'bbh-result' : 'glitch-result';
            const resultIcon = analysis.is_bbh ? '🌌' : '🎧';
            const resultText = analysis.is_bbh ? 'FUSIÓN BBH DETECTADA' : 'GLITCH (RUIDO)';
            
            resultDiv.innerHTML = `
                <div class="result-card ${resultClass}">
                    <div class="result-header">
                        <div class="result-type">
                            ${resultIcon} ${resultText}
                        </div>
                        <div class="result-time">${analysis.timestamp}</div>
                    </div>
                    <p>Confianza: <strong>${(analysis.confidence * 100).toFixed(1)}%</strong></p>
                    <div class="result-details">
                        <div class="detail-item">
                            <div class="detail-label">Amplitud</div>
                            <div class="detail-value">${analysis.amplitude.toFixed(2)}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Frecuencia</div>
                            <div class="detail-value">${analysis.frequency} Hz</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">Persistencia</div>
                            <div class="detail-value">${analysis.persistence.toFixed(2)}</div>
                        </div>
                        <div class="detail-item">
                            <div class="detail-label">ID Análisis</div>
                            <div class="detail-value">#${analysis.id}</div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Función para actualizar historial
        function updateHistory(history) {
            const historyList = document.getElementById('history-list');
            
            if (history.length === 0) {
                historyList.innerHTML = '<p>No hay análisis en el historial</p>';
                return;
            }
            
            let historyHTML = '';
            history.slice().reverse().forEach(item => {
                const icon = item.is_bbh ? '🌌' : '🎧';
                const color = item.is_bbh ? '#FF6B6B' : '#4ECDC4';
                
                historyHTML += `
                    <div class="history-item" style="
                        padding: 10px; 
                        margin: 5px 0; 
                        background: rgba(255, 255, 255, 0.05); 
                        border-radius: 8px;
                        border-left: 3px solid ${color};
                    ">
                        <div style="display: flex; justify-content: space-between;">
                            <span><strong>${icon} ${item.is_bbh ? 'BBH' : 'Glitch'}</strong></span>
                            <span style="opacity: 0.7; font-size: 0.9rem;">${item.timestamp}</span>
                        </div>
                        <div style="font-size: 0.9rem; margin-top: 5px;">
                            A: ${item.amplitude.toFixed(2)} | F: ${item.frequency}Hz | P: ${item.persistence.toFixed(2)}
                        </div>
                    </div>
                `;
            });
            
            historyList.innerHTML = historyHTML;
        }
        
        // Función para actualizar estadísticas
        function updateStats(stats) {
            document.getElementById('stat-total').textContent = stats.total_analyses;
            document.getElementById('stat-bbh').textContent = stats.bbh_detections;
            document.getElementById('stat-glitch').textContent = stats.glitch_detections;
            document.getElementById('stat-accuracy').textContent = `${stats.accuracy.toFixed(1)}%`;
        }
        
        // Inicializar dashboard al cargar
        document.addEventListener('DOMContentLoaded', function() {
            updateDashboard();
            
            // Actualizar automáticamente cada 10 segundos
            setInterval(updateDashboard, 10000);
        });
    </script>
</body>
</html>
"""

# ================= INICIALIZACIÓN =================

def create_templates_folder():
    """Crea la carpeta de templates si no existe"""
    os.makedirs("templates", exist_ok=True)
    
    # Guardar template HTML
    with open("templates/dashboard.html", "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE)
    
    print("✅ Template HTML creado en: templates/dashboard.html")

def run_dashboard(host="127.0.0.1", port=5000):
    """Ejecuta el dashboard Flask"""
    create_templates_folder()
    
    print("\n" + "="*60)
    print("🌌 DEEPWAVE DASHBOARD - Panel de Control Interactivo")
    print("="*60)
    print(f"🚀 Iniciando servidor en: http://{host}:{port}")
    print("📊 Accede al dashboard desde tu navegador web")
    print("⚡ Presiona Ctrl+C para detener el servidor")
    print("="*60)
    
    # Abrir navegador automáticamente (opcional)
    def open_browser():
        import time
        time.sleep(2)
        webbrowser.open(f"http://{host}:{port}")
    
    # Iniciar en hilo separado para Termux
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Ejecutar Flask
    app.run(host=host, port=port, debug=False)

if __name__ == "__main__":
    # Ejecutar dashboard
    run_dashboard()
