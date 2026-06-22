import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'codigo_fuente'))
from deepwave_classifier_cnn import DeepWaveCNN
from deepwave_preprocessing import generar_senal_bbh, calcular_espectrograma_stub

def test_simular_forward_pass():
    cnn = DeepWaveCNN()
    senal, fs = generar_senal_bbh()
    spec = calcular_espectrograma_stub(senal, fs)
    clase, prob = cnn.simular_forward_pass(spec)
    assert clase in (0, 1)
    assert 0.0 <= prob <= 1.0
