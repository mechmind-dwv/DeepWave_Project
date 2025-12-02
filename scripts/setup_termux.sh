#!/bin/bash
# scripts/setup_termux.sh - Configuración optimizada para Termux

echo "📱 CONFIGURADOR DEEPWAVE PARA TERMUX"
echo "===================================="

# 1. Verificar Termux
echo "1. Verificando entorno Termux..."
if [ ! -d "/data/data/com.termux" ]; then
    echo "⚠️  No parece ser Termux. Usando método estándar..."
    python3 -m venv venv_deepwave
    source venv_deepwave/bin/activate
else
    echo "✅ Entorno Termux detectado"
    # En Termux, usamos el Python del sistema con --user
    pip install --user --upgrade pip
fi

# 2. Instalar dependencias con método seguro
echo "2. Instalando dependencias Python..."
REQUIREMENTS="codigo_fuente/requirements.txt"

if [ -f "$REQUIREMENTS" ]; then
    echo "📦 Usando requirements.txt del proyecto"
    
    # Versión ligera para Termux
    cat > requirements_termux.txt << 'EOF'
numpy==1.24.3  # Versión compatible con Termux
scipy==1.10.1
matplotlib==3.7.1
EOF
    
    pip install --user -r requirements_termux.txt
    
    # Instalar TensorFlow lite (alternativa ligera)
    echo "📊 Instalando TensorFlow Lite (alternativa ligera)..."
    pip install --user tflite-runtime
    
else
    echo "📦 Instalando paquetes básicos..."
    pip install --user numpy scipy matplotlib
fi

# 3. Configurar estructura del proyecto
echo "3. Configurando estructura del proyecto..."

# Crear .gitignore específico
cat > .gitignore << 'EOF'
# Termux
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/

# Datos
data/raw/
models/*.h5
*.npy
*.npz

# Logs y temporales
*.log
*.tmp
*.temp

# IDE
.vscode/
.idea/
*.swp
*.swo

# Sistema
.DS_Store
Thumbs.db
EOF

# 4. Crear script de ejecución simplificado
cat > run_deepwave.py << 'EOF'
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
EOF

chmod +x run_deepwave.py

# 5. Crear verificación de dependencias
cat > check_deps.py << 'EOF'
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
EOF

chmod +x check_deps.py

echo ""
echo "✅ Configuración para Termux completada!"
echo ""
echo "📋 PASOS SIGUIENTES:"
echo "1. Instalar dependencias del sistema:"
echo "   pkg install python-numpy python-scipy matplotlib"
echo ""
echo "2. Verificar instalación:"
echo "   python check_deps.py"
echo ""
echo "3. Ejecutar DeepWave:"
echo "   python run_deepwave.py"
echo ""
echo "4. Si necesitas más paquetes:"
echo "   pip install --user <paquete>"
