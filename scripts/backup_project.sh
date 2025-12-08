#!/bin/bash
# scripts/backup_project.sh - Backup seguro del proyecto DeepWave

echo "💾 BACKUP SEGURO DEEPWAVE PROJECT"
echo "================================"

# Crear carpeta de backups si no existe
BACKUP_DIR="../DeepWave_Backups"
mkdir -p "$BACKUP_DIR"

# Fecha y hora para el nombre del backup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="DeepWave_Backup_${TIMESTAMP}"
BACKUP_PATH="${BACKUP_DIR}/${BACKUP_NAME}"

echo "1. Creando backup en: $BACKUP_PATH"

# Copiar todo excepto cache y archivos temporales
rsync -av --progress \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='venv_*' \
    --exclude='.git' \
    --exclude='FETCH_HEAD' \
    . "$BACKUP_PATH/"

# Crear archivo comprimido adicional
echo "2. Creando archivo comprimido..."
tar -czf "${BACKUP_PATH}.tar.gz" -C "$(dirname "$BACKUP_PATH")" "$(basename "$BACKUP_PATH")"

# Verificar el backup
echo "3. Verificando backup..."
if [ -d "$BACKUP_PATH" ]; then
    echo "✅ Backup de código fuente:"
    find "$BACKUP_PATH" -name "*.py" | wc -l | xargs echo "   - Archivos .py:"
    du -sh "$BACKUP_PATH" | cut -f1 | xargs echo "   - Tamaño total:"
    
    echo ""
    echo "📦 Archivos críticos protegidos:"
    ls "$BACKUP_PATH/codigo_fuente/"*.py 2>/dev/null | xargs -I {} basename {} | while read f; do
        echo "   ✅ $f"
    done
else
    echo "❌ Error en backup"
    exit 1
fi

echo ""
echo "🎉 BACKUP COMPLETADO EXITOSAMENTE"
echo "📍 Ubicación: $BACKUP_PATH"
echo "📦 Comprimido: ${BACKUP_PATH}.tar.gz"
echo ""
echo "🔒 Tu proyecto mágico está seguro. Puedes proceder con limpieza."
