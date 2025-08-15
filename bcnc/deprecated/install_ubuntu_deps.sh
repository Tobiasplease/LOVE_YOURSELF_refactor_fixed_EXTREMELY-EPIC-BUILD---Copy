#!/bin/bash
# Installation script for bCNC GUI Controller dependencies on Ubuntu

echo "Installing bCNC GUI Controller dependencies for Ubuntu..."

# Update package list
echo "Updating package list..."
sudo apt update

# Install system dependencies for window management
echo "Installing window management tools..."
sudo apt install -y xdotool wmctrl x11-utils

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements_bcnc_ubuntu.txt

# Check if bCNC is available
if ! command -v bCNC &> /dev/null; then
    echo "bCNC not found. You may need to install it separately:"
    echo "Option 1: sudo apt install bcnc (if available in your distribution)"
    echo "Option 2: Download from https://github.com/vlachoudis/bCNC"
    echo "Option 3: pip install bCNC"
fi

echo "Installation complete!"
echo ""
echo "Usage:"
echo "  python bcnc_gui_controller_ubuntu.py"
echo ""
echo "Make sure bCNC is running before using the controller:"
echo "bCNC if installed via apt."