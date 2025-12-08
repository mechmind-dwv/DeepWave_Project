#!/bin/bash
# scripts/setup_deepwave.sh - Configuración especializada para DeepWave Project

echo "🌌 CONFIGURADOR DEEPWAVE PROJECT"
echo "================================="

# 1. Configurar entorno Python
echo "1. Configurando entorno Python..."
python3 -m venv venv_deepwave
source venv_deepwave/bin/activate  # En Linux/Mac
# Para Windows: venv_deepwave\Scripts\activate

pip install numpy scipy matplotlib tensorflow keras h5py

# 2. Verificar estructura del proyecto
echo "2. Verificando estructura..."
if [ ! -d "codigo_fuente" ]; then
    echo "ERROR: No se encuentra la carpeta 'codigo_fuente'"
    exit 1
fi

# 3. Crear archivos de configuración
echo "3. Creando archivos de configuración..."

# .gitignore para Python/ML
cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv_deepwave/
env/

# Datos y modelos
data/raw/
models/*.h5
*.npy
*.npz

# IDE
.vscode/
.idea/
*.swp
*.swo

# Sistema
.DS_Store
Thumbs.db
EOF

# requirements.txt
cat > requirements.txt << EOF
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.4.0
tensorflow>=2.6.0
keras>=2.6.0
h5py>=3.1.0
EOF

echo "✅ Configuración completada!"
echo ""
echo "Comandos útiles:"
echo "  source venv_deepwave/bin/activate  # Activar entorno virtual"
echo "  python codigo_fuente/deepwave_core.py  # Ejecutar módulo principal"
echo "  git add . && git commit -m 'update' && git push  # Subir cambios"
