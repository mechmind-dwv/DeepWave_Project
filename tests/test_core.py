import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'codigo_fuente'))
from deepwave_core import DeepWaveClassifier

def test_prediccion_bbh():
    detector = DeepWaveClassifier()
    resultado = detector.predecir_evento([0.1, 50, 0.9])
    assert "FUSIÓN BBH" in resultado

def test_prediccion_glitch():
    detector = DeepWaveClassifier()
    resultado = detector.predecir_evento([0.9, 150, 0.2])
    assert "GLITCH" in resultado
