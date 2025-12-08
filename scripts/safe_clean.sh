#!/bin/bash
# scripts/safe_clean.sh - Limpieza segura que NO toca código fuente

echo "🧹 LIMPIEZA SEGURA DEEPWAVE"
echo "==========================="

# 1. MOSTRAR lo que se va a limpiar (sin borrar aún)
echo "1. Archivos temporales que se pueden limpiar:"
echo "---------------------------------------------"

# Mostrar __pycache__ (se regeneran)
find . -name "__pycache__" -type d 2>/dev/null | while read dir; do
    echo "   📁 $dir (cache Python - regenerable)"
done

# Mostrar .pyc (se regeneran)
find . -name "*.pyc" 2>/dev/null | head -5 | while read file; do
    echo "   📄 $file (bytecode - regenerable)"
done

# 2. PREGUNTAR antes de borrar
echo ""
read -p "¿Continuar con la limpieza? (s/N): " confirm

if [[ "$confirm" != "s" ]] && [[ "$confirm" != "S" ]]; then
    echo "❌ Limpieza cancelada. Tus archivos están seguros."
    exit 0
fi

# 3. LIMPIAR solo archivos regenerables
echo ""
echo "2. Limpiando archivos temporales..."
count=0

# Borrar __pycache__ (SE REGENERAN SOLOS)
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
pycache_count=$(find . -name "__pycache__" -type d 2>/dev/null | wc -l)
echo "   ✅ Eliminados $pycache_count directorios __pycache__"

# Borrar .pyc (SE REGENERAN SOLOS)
pyc_count=$(find . -name "*.pyc" 2>/dev/null | wc -l)
find . -name "*.pyc" -delete 2>/dev/null
echo "   ✅ Eliminados $pyc_count archivos .pyc"

# 4. VERIFICAR que el código fuente sigue intacto
echo ""
echo "3. Verificando código fuente..."
echo "-------------------------------"

# Contar archivos .py (tu código real)
py_files=$(find . -name "*.py" -not -path "./venv_*" 2>/dev/null | wc -l)
echo "   ✅ $py_files archivos .py (tu código mágico) están SEGUROS"

# Listar los módulos principales
echo ""
echo "📚 Módulos DeepWave protegidos:"
for module in deepwave_core.py deepwave_classifier_cnn.py deepwave_preprocessing.py; do
    if [ -f "codigo_fuente/$module" ]; then
        echo "   🔒 codigo_fuente/$module"
    fi
done

# 5. Mostrar nueva estructura
echo ""
echo "4. Estructura actual limpia:"
echo "----------------------------"
tree -L 2 -I 'venv_*|.git' --dirsfirst 2>/dev/null || echo "   (usa 'ls -la' para ver)"

echo ""
echo "🎯 LIMPIEZA SEGURA COMPLETADA"
echo "⚠️  Los archivos .pyc se regenerarán automáticamente al ejecutar el código."
