#!/bin/bash
# Wave Global Uninstaller Script
# Removes Wave global 'wave' command

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if we're on macOS or Linux
OS_TYPE=$(uname)

# Determine where the symlink should be installed
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

echo -e "${BLUE}🌊 Wave Global Uninstaller${NC}"
echo -e "${BLUE}===========================${NC}"
echo ""

WAVE_COMMAND="$INSTALL_DIR/wave"

# Check if wave command exists
if [[ ! -e "$WAVE_COMMAND" ]]; then
    echo -e "${YELLOW}⚠️  Wave command not found at: $WAVE_COMMAND${NC}"
    echo "Wave may not be globally installed or may be installed elsewhere."
    
    # Check if wave is in PATH but from a different location
    if command -v wave >/dev/null 2>&1; then
        ACTUAL_LOCATION=$(which wave)
        echo -e "${YELLOW}💡 Found wave command at: $ACTUAL_LOCATION${NC}"
        echo "This script only removes installations from $INSTALL_DIR"
        echo "If you want to remove the installation from $ACTUAL_LOCATION, please do so manually."
    fi
    exit 0
fi

# Check if it's actually a symlink to Wave (safety check)
if [[ -L "$WAVE_COMMAND" ]]; then
    LINK_TARGET=$(readlink "$WAVE_COMMAND")
    echo "Found Wave symlink: $WAVE_COMMAND -> $LINK_TARGET"
    
    # Verify it's actually pointing to a Wave installation
    if [[ "$LINK_TARGET" == *"wave"* ]] && [[ -f "$LINK_TARGET" ]]; then
        echo -e "${GREEN}✅ Confirmed this is a Wave installation${NC}"
    else
        echo -e "${YELLOW}⚠️  This symlink doesn't appear to point to Wave. Proceeding anyway...${NC}"
    fi
elif [[ -f "$WAVE_COMMAND" ]]; then
    echo -e "${YELLOW}⚠️  Found a file (not symlink) at $WAVE_COMMAND${NC}"
    echo "This may not be a Wave installation created by the installer script."
    echo "Checking if it's a Wave script..."
    
    # Check if it's a Wave script by looking for the header
    if head -5 "$WAVE_COMMAND" | grep -q "Wave.*Trading Bot" 2>/dev/null; then
        echo -e "${GREEN}✅ This appears to be a Wave script${NC}"
    else
        echo -e "${RED}❌ This doesn't appear to be a Wave installation${NC}"
        echo "Aborting for safety. Please remove manually if needed."
        exit 1
    fi
else
    echo -e "${RED}❌ Unexpected file type at $WAVE_COMMAND${NC}"
    exit 1
fi

# Check if we need sudo
NEED_SUDO=false
if [[ ! -w "$INSTALL_DIR" ]]; then
    NEED_SUDO=true
    echo -e "${YELLOW}⚠️  Need sudo privileges to remove from $INSTALL_DIR${NC}"
fi

# Confirm removal
echo ""
echo -e "${YELLOW}❓ Are you sure you want to remove Wave? (y/N)${NC}"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "❌ Uninstall cancelled."
    exit 0
fi

# Remove the wave command
echo -e "${YELLOW}🗑️  Removing Wave command...${NC}"
if [[ "$NEED_SUDO" == "true" ]]; then
    sudo rm "$WAVE_COMMAND"
else
    rm "$WAVE_COMMAND"
fi

# Verify removal
if [[ ! -e "$WAVE_COMMAND" ]]; then
    echo -e "${GREEN}✅ Wave command removed successfully!${NC}"
    echo ""
    echo "Wave has been uninstalled from your system."
    echo "The Wave project directory and its contents remain unchanged."
    echo ""
    echo -e "${BLUE}💡 Note:${NC} If you want to remove the project directory as well, you can:"
    echo "   rm -rf $(dirname "${BASH_SOURCE[0]}")"
    echo ""
    echo -e "${BLUE}💡 To reinstall Wave later, run:${NC}"
    echo "   ./install.sh"
else
    echo -e "${RED}❌ Failed to remove Wave command. Please check permissions and try again.${NC}"
    exit 1
fi

echo -e "${GREEN}🎉 Wave uninstallation complete!${NC}"