#!/bin/bash

# Colores para mensajes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para mostrar mensajes
print_message() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[ÉXITO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[ADVERTENCIA]${NC} $1"
}

# Verificar si Git está instalado
check_git_installed() {
    if ! command -v git &> /dev/null; then
        print_error "Git no está instalado."
        echo "Instala Git primero:"
        echo "  Ubuntu/Debian: sudo apt install git"
        echo "  macOS: brew install git"
        echo "  Windows: Descarga de git-scm.com"
        exit 1
    fi
    print_success "Git está instalado: $(git --version)"
}

# Configurar usuario de Git
configure_git_user() {
    print_message "Configurando usuario de Git..."
    
    current_name=$(git config --global user.name)
    current_email=$(git config --global user.email)
    
    if [ -n "$current_name" ] && [ -n "$current_email" ]; then
        echo "Configuración actual:"
        echo "  Nombre:  $current_name"
        echo "  Email:   $current_email"
        read -p "¿Quieres cambiar la configuración? (s/n): " change_config
        if [[ $change_config != "s" ]] && [[ $change_config != "S" ]]; then
            return
        fi
    fi
    
    read -p "Tu nombre completo (para commits): " user_name
    read -p "Tu email (de GitHub): " user_email
    
    git config --global user.name "$user_name"
    git config --global user.email "$user_email"
    
    print_success "Usuario Git configurado: $user_name <$user_email>"
}

# Método 1: Configurar Token PAT (HTTPS)
configure_pat_token() {
    print_message "Configurando Token PAT para HTTPS..."
    
    echo ""
    echo "Para crear un Token PAT (Personal Access Token):"
    echo "1. Ve a: https://github.com/settings/tokens"
    echo "2. Haz clic en 'Generate new token'"
    echo "3. Selecciona 'repo' (para todo el control de repositorios)"
    echo "4. También selecciona 'workflow' si usas GitHub Actions"
    echo "5. Genera el token y cópialo (aparecerá solo una vez)"
    echo ""
    
    read -p "¿Ya tienes un token PAT listo? (s/n): " has_token
    if [[ $has_token != "s" ]] && [[ $has_token != "S" ]]; then
        print_warning "Crea el token primero y luego vuelve a ejecutar el script."
        return
    fi
    
    read -sp "Pega tu Token PAT aquí: " pat_token
    echo ""
    
    if [ -z "$pat_token" ]; then
        print_error "No ingresaste ningún token."
        return
    fi
    
    # Guardar token en el almacén de credenciales de Git
    echo "https://$pat_token@github.com" | git credential-store --file ~/.git-credentials store
    
    # Configurar Git para usar el almacén de credenciales
    git config --global credential.helper "store --file ~/.git-credentials"
    
    print_success "Token PAT configurado correctamente."
    print_warning "El token se guardó en ~/.git-credentials"
    print_warning "Guarda este archivo de forma segura y no lo compartas."
    
    # Probar la conexión
    print_message "Probando conexión con GitHub..."
    if git ls-remote https://github.com/mechmind-dwv/DeepWave_Project.git &> /dev/null; then
        print_success "¡Conexión HTTPS exitosa!"
    else
        print_error "No se pudo conectar. Verifica tu token."
    fi
}

# Método 2: Configurar SSH
configure_ssh() {
    print_message "Configurando SSH para GitHub..."
    
    # Verificar si ya existe una clave SSH
    if [ -f ~/.ssh/id_ed25519 ] || [ -f ~/.ssh/id_rsa ]; then
        print_message "Se encontraron claves SSH existentes:"
        ls -la ~/.ssh/id_* 2>/dev/null || echo "  No se encontraron claves específicas"
        
        read -p "¿Quieres generar una nueva clave SSH? (s/n): " generate_new
        if [[ $generate_new != "s" ]] && [[ $generate_new != "S" ]]; then
            print_message "Usando claves existentes..."
        else
            generate_ssh_key
        fi
    else
        generate_ssh_key
    fi
    
    # Iniciar el agente SSH y añadir la clave
    print_message "Iniciando agente SSH..."
    eval "$(ssh-agent -s)" > /dev/null 2>&1
    
    # Intentar añadir la clave Ed25519 primero, luego RSA
    if [ -f ~/.ssh/id_ed25519 ]; then
        ssh-add ~/.ssh/id_ed25519 2>/dev/null
        ssh_key_type="ed25519"
    elif [ -f ~/.ssh/id_rsa ]; then
        ssh-add ~/.ssh/id_rsa 2>/dev/null
        ssh_key_type="rsa"
    else
        print_error "No se encontró ninguna clave SSH válida."
        return
    fi
    
    # Mostrar la clave pública
    print_success "Clave SSH ($ssh_key_type) configurada."
    echo ""
    print_message "Tu clave pública SSH es:"
    echo "---------------------------------------------------"
    cat ~/.ssh/id_${ssh_key_type}.pub
    echo "---------------------------------------------------"
    echo ""
    
    print_message "Para configurar GitHub con SSH:"
    echo "1. Ve a: https://github.com/settings/keys"
    echo "2. Haz clic en 'New SSH key'"
    echo "3. Pega el contenido anterior en el campo 'Key'"
    echo "4. Dale un título descriptivo"
    echo "5. Haz clic en 'Add SSH key'"
    echo ""
    
    read -p "¿Ya añadiste la clave SSH a GitHub? (s/n): " key_added
    if [[ $key_added != "s" ]] && [[ $key_added != "S" ]]; then
        print_warning "Añade la clave SSH a GitHub para continuar."
        return
    fi
    
    # Probar la conexión SSH
    print_message "Probando conexión SSH con GitHub..."
    if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
        print_success "¡Conexión SSH exitosa!"
    else
        ssh -T git@github.com
        print_warning "La conexión SSH podría necesitar configuración adicional."
    fi
}

# Función para generar nueva clave SSH
generate_ssh_key() {
    print_message "Generando nueva clave SSH..."
    
    echo ""
    echo "Se recomienda usar Ed25519 (más seguro y rápido)."
    echo "Opciones:"
    echo "  1) Ed25519 (Recomendado)"
    echo "  2) RSA 4096 bits (Compatible con sistemas antiguos)"
    echo ""
    
    read -p "Elige el tipo de clave (1 o 2): " key_choice
    
    read -p "Tu email de GitHub: " ssh_email
    
    case $key_choice in
        2)
            print_message "Generando clave RSA 4096 bits..."
            ssh-keygen -t rsa -b 4096 -C "$ssh_email" -f ~/.ssh/id_rsa -N ""
            ;;
        *)
            print_message "Generando clave Ed25519..."
            ssh-keygen -t ed25519 -C "$ssh_email" -f ~/.ssh/id_ed25519 -N ""
            ;;
    esac
    
    print_success "Clave SSH generada correctamente."
}

# Configurar el repositorio remoto
configure_remote_repo() {
    print_message "Configurando repositorio remoto..."
    
    read -p "URL de tu repositorio GitHub [https://github.com/mechmind-dwv/DeepWave_Project.git]: " repo_url
    repo_url=${repo_url:-"https://github.com/mechmind-dwv/DeepWave_Project.git"}
    
    # Verificar si ya existe un remote 'origin'
    if git remote get-url origin &> /dev/null; then
        current_url=$(git remote get-url origin)
        print_message "Remote 'origin' ya existe: $current_url"
        read -p "¿Quieres cambiarlo a $repo_url? (s/n): " change_remote
        
        if [[ $change_remote == "s" ]] || [[ $change_remote == "S" ]]; then
            git remote set-url origin "$repo_url"
            print_success "Remote 'origin' actualizado."
        fi
    else
        git remote add origin "$repo_url"
        print_success "Remote 'origin' añadido."
    fi
    
    # Si es SSH, cambiar la URL a formato SSH
    if [[ $auth_method == "2" ]] && [[ $repo_url == https://* ]]; then
        # Convertir HTTPS a SSH
        ssh_url=$(echo "$repo_url" | sed 's|https://github.com/|git@github.com:|')
        git remote set-url origin "$ssh_url"
        print_success "URL cambiada a formato SSH: $ssh_url"
    fi
}

# Menú principal
main_menu() {
    clear
    echo "========================================="
    echo "  CONFIGURADOR GITHUB - DeepWave Project  "
    echo "========================================="
    echo ""
    
    check_git_installed
    echo ""
    
    configure_git_user
    echo ""
    
    echo "Métodos de autenticación disponibles:"
    echo "  1) Token PAT (Personal Access Token) - HTTPS"
    echo "     - Fácil de usar"
    echo "     - Necesita token por cada repositorio"
    echo ""
    echo "  2) SSH"
    echo "     - Más seguro"
    echo "     - Configuración única por máquina"
    echo "     - Recomendado para desarrollo"
    echo ""
    
    read -p "Elige método de autenticación (1 o 2): " auth_method
    
    case $auth_method in
        1)
            configure_pat_token
            ;;
        2)
            configure_ssh
            ;;
        *)
            print_error "Opción no válida. Saliendo."
            exit 1
            ;;
    esac
    
    echo ""
    
    # Configurar repositorio remoto
    if [ -d .git ]; then
        configure_remote_repo
    else
        print_warning "Esta carpeta no es un repositorio Git."
        read -p "¿Quieres inicializarla como repositorio Git? (s/n): " init_repo
        if [[ $init_repo == "s" ]] || [[ $init_repo == "S" ]]; then
            git init
            print_success "Repositorio Git inicializado."
            configure_remote_repo
        fi
    fi
    
    echo ""
    echo "========================================="
    print_success "Configuración completada"
    echo ""
    
    # Mostrar resumen
    print_message "Resumen de configuración:"
    echo "  Nombre:  $(git config --global user.name)"
    echo "  Email:   $(git config --global user.email)"
    echo "  Remote:  $(git remote get-url origin 2>/dev/null || echo 'No configurado')"
    echo ""
    
    # Prueba final
    read -p "¿Quieres probar el push a GitHub? (s/n): " test_push
    if [[ $test_push == "s" ]] || [[ $test_push == "S" ]]; then
        print_message "Haciendo prueba de push..."
        
        # Crear un README de prueba si no existe
        if [ ! -f README.md ]; then
            echo "# DeepWave Project" > README.md
            echo "Configurado el $(date)" >> README.md
            git add README.md
            git commit -m "Añade README desde script de configuración" --quiet
        fi
        
        if git push -u origin main 2>/dev/null || git push -u origin master 2>/dev/null; then
            print_success "¡Push exitoso! Tu configuración funciona correctamente."
        else
            print_warning "El push falló. Verifica tu configuración."
        fi
    fi
    
    echo ""
    print_message "Comandos útiles para tu proyecto:"
    echo "  git add .                        # Añadir todos los cambios"
    echo "  git commit -m \"mensaje\"         # Guardar cambios con mensaje"
    echo "  git push                         # Subir cambios a GitHub"
    echo "  git pull                         # Descargar cambios de GitHub"
    echo "  git status                       # Ver estado actual"
    echo ""
}

# Ejecutar menú principal
main_menu
