#!/usr/bin/env python3
"""
Script de ejecución principal para DeepWave en Termux
"""

import sys
import os

# Añadir ruta de usuario de Termux a sys.path
termux_path = "/data/data/com.termux/files/home/.local/lib/python3.11/site-packages"
if os.path.exists(termux_path) and termux_path not in sys.path:
    sys.path.insert(0, termux_path)

print("🌌 DeepWave Project - Sistema de Detección BBH")
print("============================================")

# Listar módulos disponibles
modules = {
    "1": "deepwave_core.py",
    "2": "deepwave_preprocessing.py", 
    "3": "deepwave_classifier_cnn.py",
    "4": "Todos los módulos (test completo)"
}

for key, value in modules.items():
    print(f"  {key}. {value}")

try:
    choice = input("\nSelecciona módulo a ejecutar (1-4): ").strip()
    
    if choice == "1":
        print("\n🔍 Ejecutando deepwave_core.py...")
        import codigo_fuente.deepwave_core
    elif choice == "2":
        print("\n⚙️  Ejecutando deepwave_preprocessing.py...")
        import codigo_fuente.deepwave_preprocessing
    elif choice == "3":
        print("\n🧠 Ejecutando deepwave_classifier_cnn.py...")
        import codigo_fuente.deepwave_classifier_cnn
    elif choice == "4":
        print("\n🚀 Ejecutando test completo...")
        import codigo_fuente.deepwave_core
        print("-" * 40)
        import codigo_fuente.deepwave_preprocessing
        print("-" * 40)
        import codigo_fuente.deepwave_classifier_cnn
    else:
        print("❌ Opción no válida")
        
except ImportError as e:
    print(f"\n❌ Error de importación: {e}")
    print("\n💡 Solución: Ejecuta primero:")
    print("  pip install --user numpy scipy matplotlib")
    print("  o ejecuta: ./scripts/setup_termux.sh")
except Exception as e:
    print(f"\n⚠️  Error durante la ejecución: {e}")

print("\n✅ Proceso completado")
