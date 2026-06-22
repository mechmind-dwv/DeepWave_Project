#!/usr/bin/env python3
"""
🌌 DEEPWAVE - Sistema de Detección de Fusión BBH
Ejecución principal optimizada para Termux
"""

import sys
import os
import time
import subprocess
from datetime import datetime

# ================= CONFIGURACIÓN DE PATHS =================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, "codigo_fuente"))

# ================= IMPORTACIÓN SEGURA =================
def safe_import(module_name, class_name=None):
    """Importa un módulo de forma segura"""
    try:
        if module_name == "deepwave_core":
            from codigo_fuente import deepwave_core as module
            print("✅ Módulo core importado exitosamente")
        elif module_name == "deepwave_preprocessing":
            from codigo_fuente import deepwave_preprocessing as module
            print("✅ Preprocesamiento importado")
        elif module_name == "deepwave_classifier_cnn":
            from codigo_fuente import deepwave_classifier_cnn as module
            print("✅ Clasificador CNN importado")
        else:
            print(f"❌ Módulo {module_name} no reconocido")
            return None

        if class_name:
            return getattr(module, class_name, None)
        return module

    except ImportError as e:
        print(f"❌ Error importando {module_name}: {e}")
        print(f"💡 Asegúrate de que 'codigo_fuente/{module_name}.py' existe")
        return None
    except Exception as e:
        print(f"⚠️  Error inesperado con {module_name}: {e}")
        return None

# ================= FUNCIONES PRINCIPALES =================
def print_header():
    """Imprime el encabezado del sistema"""
    print("\n" + "="*60)
    print("🌌 DEEPWAVE: Sistema de Detección de Fusión BBH")
    print("🧠 IA para Análisis de Ondas Gravitacionales")
    print("="*60)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python {sys.version.split()[0]}")
    print("="*60)

def check_dependencies():
    """Verifica dependencias críticas"""
    print("\n🔍 Verificando dependencias...")

    deps = {
        "NumPy": "numpy",
        "SciPy": "scipy",
        "Matplotlib": "matplotlib",
    }

    missing = []
    for name, module in deps.items():
        try:
            __import__(module)
            print(f"   ✅ {name:15} ... OK")
        except ImportError:
            print(f"   ❌ {name:15} ... FALTANTE")
            missing.append(module)

    return missing

def run_module(module_name, description):
    """Ejecuta un módulo específico"""
    print(f"\n🚀 {description}")
    print("-" * 40)

    start_time = time.time()

    try:
        if module_name == "core":
            subprocess.run([sys.executable, os.path.join("codigo_fuente", "deepwave_core.py")], check=True)
            print("✅ Módulo core ejecutado exitosamente")

        elif module_name == "preprocessing":
            subprocess.run([sys.executable, os.path.join("codigo_fuente", "deepwave_preprocessing.py")], check=True)
            print("✅ Preprocesamiento completado")

        elif module_name == "classifier":
            # Intentar usar CNN real si el modelo entrenado existe
            try:
                from codigo_fuente.deepwave_classifier_cnn_real import RealDeepWaveCNN
                cnn = RealDeepWaveCNN("models/best_cnn.h5")
                print("✅ CNN real cargada. Realizando predicción de prueba...")
                from codigo_fuente.deepwave_preprocessing import generar_senal_bbh, calcular_espectrograma_stub
                senal, fs = generar_senal_bbh()
                spec = calcular_espectrograma_stub(senal, fs)
                clase, prob = cnn.predict(spec)
                resultado = "FUSIÓN BBH 🌌" if clase == 1 else "GLITCH 🎧"
                print(f"  -> Resultado: {resultado}")
                print(f"  -> Probabilidad BBH: {prob:.4f}")
            except Exception as e:
                print(f"⚠️  No se pudo cargar la CNN real ({e})")
                print("Ejecutando CNN simulada (demostración)...")
                subprocess.run([sys.executable, os.path.join("codigo_fuente", "deepwave_classifier_cnn.py")], check=True)
            print("✅ Clasificador CNN ejecutado")

        else:
            print("❌ Módulo no reconocido")
            return False

        elapsed = time.time() - start_time
        print(f"⏱️  Tiempo de ejecución: {elapsed:.2f} segundos")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {module_name} (código {e.returncode})")
        return False
    except Exception as e:
        print(f"❌ Error en {module_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main_menu():
    """Menú principal interactivo"""
    while True:
        print("\n📋 MENÚ PRINCIPAL DEEPWAVE")
        print("1. Ejecutar análisis completo (todos los módulos)")
        print("2. Ejecutar solo clasificador K-NN (core)")
        print("3. Ejecutar preprocesamiento STFT")
        print("4. Ejecutar clasificador CNN")
        print("5. Verificar dependencias")
        print("6. Mostrar estructura del proyecto")
        print("7. Salir")

        try:
            choice = input("\n👉 Selecciona una opción (1-7): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 ¡Hasta pronto!")
            break

        if choice == "1":
            print("\n🔬 INICIANDO ANÁLISIS COMPLETO")
            print("=" * 40)
            run_module("core", "1. Clasificación inicial K-NN")
            run_module("preprocessing", "2. Transformada STFT")
            run_module("classifier", "3. Clasificación profunda CNN")
            print("\n🎉 ANÁLISIS COMPLETADO EXITOSAMENTE")

        elif choice == "2":
            run_module("core", "Ejecutando clasificador K-NN")

        elif choice == "3":
            run_module("preprocessing", "Ejecutando preprocesamiento STFT")

        elif choice == "4":
            run_module("classifier", "Ejecutando clasificador CNN")

        elif choice == "5":
            missing = check_dependencies()
            if missing:
                print(f"\n⚠️  Instala dependencias faltantes:")
                print(f"   pkg install python-{' python-'.join(missing)}")
            else:
                print("\n✅ Todas las dependencias están instaladas")

        elif choice == "6":
            print("\n📁 ESTRUCTURA DEL PROYECTO:")
            try:
                os.system("tree -L 2 -I '__pycache__|*.pyc|venv*' --dirsfirst 2>/dev/null")
            except:
                print("(usa 'ls -la' para ver)")

        elif choice == "7":
            print("\n👋 ¡Hasta pronto! Que las ondas gravitacionales te acompañen 🌌")
            break

        else:
            print("❌ Opción no válida. Intenta nuevamente.")

if __name__ == "__main__":
    print_header()

    # Verificar que estamos en el directorio correcto
    if not os.path.exists("codigo_fuente"):
        print("⚠️  No se encuentra 'codigo_fuente'. ¿Estás en el directorio correcto?")
        print("💡 Ejecuta desde: ~/DeepWave_Project")
        sys.exit(1)

    # Verificar módulos existentes
    print("\n🔍 Verificando módulos DeepWave...")
    modules = ["deepwave_core.py", "deepwave_preprocessing.py", "deepwave_classifier_cnn.py"]
    for module in modules:
        if os.path.exists(os.path.join("codigo_fuente", module)):
            print(f"   ✅ {module:30} ... ENCONTRADO")
        else:
            print(f"   ❌ {module:30} ... NO ENCONTRADO")

    # Verificar dependencias críticas
    missing = check_dependencies()
    if missing:
        print(f"\n🚨 DEPENDENCIAS CRÍTICAS FALTANTES")
        print(f"Instala primero: pkg install python-{' python-'.join(missing)}")
        response = input("¿Continuar de todos modos? (s/N): ")
        if response.lower() != 's':
            sys.exit(1)

    # Ejecutar menú principal
    main_menu()
