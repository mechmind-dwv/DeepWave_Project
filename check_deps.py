#!/usr/bin/env python3
"""
Verificador de dependencias para DeepWave en Termux
"""

import sys

print("🔍 Verificando dependencias de DeepWave...")
print("=" * 40)

dependencies = [
    ("NumPy", "numpy"),
    ("SciPy", "scipy"), 
    ("Matplotlib", "matplotlib"),
]

missing = []
for name, module in dependencies:
    try:
        __import__(module)
        print(f"✅ {name} ... OK")
    except ImportError:
        print(f"❌ {name} ... FALTANTE")
        missing.append(module)

if missing:
    print("\n⚠️  Dependencias faltantes detectadas!")
    print(f"\n📦 Para instalar, ejecuta:")
    print(f"   pip install --user {' '.join(missing)}")
    print(f"\n🔧 O usa el script de setup:")
    print(f"   ./scripts/setup_termux.sh")
else:
    print("\n🎉 ¡Todas las dependencias están instaladas!")
    print("\n🚀 Para ejecutar DeepWave:")
    print("   python run_deepwave.py")

print("\n📊 Información del sistema:")
print(f"   Python: {sys.version}")
