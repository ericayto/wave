#!/bin/bash
# Wave Global Installer Script
# Installs Wave as a global 'wave' command

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the absolute path of the wave project directory
WAVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAVE_SCRIPT="$WAVE_DIR/wave"

# Check if we're on macOS or Linux
OS_TYPE=$(uname)

# Determine where to install the symlink
if [[ "$OS_TYPE" == "Darwin" ]]; then
    # macOS
    INSTALL_DIR="/usr/local/bin"
elif [[ "$OS_TYPE" == "Linux" ]]; then
    # Linux
    INSTALL_DIR="/usr/local/bin"
else
    echo -e "${RED}❌ Unsupported operating system: $OS_TYPE${NC}"
    exit 1
fi

echo -e "${BLUE}🌊 Wave Global Installer${NC}"
echo -e "${BLUE}========================${NC}"
echo ""
echo "Installing Wave to: $INSTALL_DIR/wave"
echo "Wave project directory: $WAVE_DIR"
echo ""

# Check if wave script exists
if [[ ! -f "$WAVE_SCRIPT" ]]; then
    echo -e "${RED}❌ Wave script not found at: $WAVE_SCRIPT${NC}"
    exit 1
fi

# Make wave script executable
echo -e "${YELLOW}📝 Making Wave script executable...${NC}"
chmod +x "$WAVE_SCRIPT"

# Check if install directory exists and is writable
if [[ ! -d "$INSTALL_DIR" ]]; then
    echo -e "${RED}❌ Install directory does not exist: $INSTALL_DIR${NC}"
    echo "Please create it with: sudo mkdir -p $INSTALL_DIR"
    exit 1
fi

# Check if we need sudo
NEED_SUDO=false
if [[ ! -w "$INSTALL_DIR" ]]; then
    NEED_SUDO=true
    echo -e "${YELLOW}⚠️  Need sudo privileges to install to $INSTALL_DIR${NC}"
fi

# Remove existing installation if it exists
EXISTING_WAVE="$INSTALL_DIR/wave"
if [[ -L "$EXISTING_WAVE" ]]; then
    echo -e "${YELLOW}🔄 Removing existing Wave installation...${NC}"
    if [[ "$NEED_SUDO" == "true" ]]; then
        sudo rm "$EXISTING_WAVE"
    else
        rm "$EXISTING_WAVE"
    fi
elif [[ -f "$EXISTING_WAVE" ]]; then
    echo -e "${YELLOW}⚠️  A file named 'wave' already exists at $EXISTING_WAVE${NC}"
    echo "Please remove it manually or choose a different installation location."
    exit 1
fi

# Create symlink
echo -e "${YELLOW}🔗 Creating symlink...${NC}"
if [[ "$NEED_SUDO" == "true" ]]; then
    sudo ln -sf "$WAVE_SCRIPT" "$EXISTING_WAVE"
else
    ln -sf "$WAVE_SCRIPT" "$EXISTING_WAVE"
fi

# Verify installation
if [[ -x "$EXISTING_WAVE" ]]; then
    echo -e "${GREEN}✅ Wave installed successfully!${NC}"
    echo ""
    echo -e "${GREEN}🚀 You can now run Wave from anywhere using: ${BLUE}wave${NC}"
    echo ""
    echo "Available commands:"
    echo "  wave setup    - Set up Wave environment"
    echo "  wave start    - Start Wave services"  
    echo "  wave stop     - Stop Wave services"
    echo "  wave update   - Update Wave to the latest version"
    echo ""
    echo -e "${YELLOW}💡 Note: Wave will auto-update on startup to ensure you have the latest features.${NC}"
else
    echo -e "${RED}❌ Installation failed. Please check permissions and try again.${NC}"
    exit 1
fi

# Test that wave is in PATH
echo -e "${YELLOW}🧪 Testing installation...${NC}"
if command -v wave >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Wave is accessible from PATH${NC}"
    
    # Show version info
    echo ""
    echo "Wave installation info:"
    echo "  Command location: $(which wave)"
    echo "  Project directory: $WAVE_DIR"
    echo "  Git repository: $(cd "$WAVE_DIR" && git remote get-url origin 2>/dev/null || echo 'Not a git repository')"
    
else
    echo -e "${RED}❌ Wave is not accessible from PATH${NC}"
    echo "You may need to restart your terminal or add $INSTALL_DIR to your PATH"
    echo ""
    echo "To add $INSTALL_DIR to your PATH, add this line to your shell profile:"
    echo "export PATH=\"$INSTALL_DIR:\$PATH\""
    exit 1
fi

echo ""
echo -e "${GREEN}🎉 Installation complete! You can now use 'wave' from anywhere.${NC}"