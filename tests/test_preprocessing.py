import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'codigo_fuente'))
from deepwave_preprocessing import generar_senal_bbh, generar_senal_glitch, calcular_espectrograma_stub
import numpy as np

def test_generar_senal_bbh_shape():
    senal, fs = generar_senal_bbh()
    assert len(senal) == 2048
    assert fs == 2048

def test_generar_senal_glitch_shape():
    senal, fs = generar_senal_glitch()
    assert len(senal) == 2048

def test_espectrograma_shape():
    senal, fs = generar_senal_bbh()
    spec = calcular_espectrograma_stub(senal, fs)
    assert spec.shape == (103, 19)
