#!/bin/bash
# Setup script for Music Stem Separator (Linux/Mac)
# Creates virtual environment and installs dependencies

set -e  # Exit on error

echo "========================================"
echo "Music Stem Separator Setup"
echo "========================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3 from your package manager or https://www.python.org/"
    exit 1
fi

echo "Python found:"
python3 --version
echo

# Check if pip is available
if ! python3 -m pip --version &> /dev/null; then
    echo "ERROR: pip is not installed"
    echo "Please install pip"
    exit 1
fi

echo "pip found:"
python3 -m pip --version
echo

# Create virtual environment
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing old virtual environment..."
        rm -rf venv
    else
        echo "Using existing virtual environment."
    fi
fi

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created successfully."
fi
echo

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install PyTorch with CUDA support (if available)
echo
echo "========================================"
echo "Installing PyTorch"
echo "========================================"
echo
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected!"
    echo "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No NVIDIA GPU detected."
    echo "Installing PyTorch (CPU-only version)..."
    pip install torch torchvision torchaudio
fi

# Install other requirements
echo
echo "========================================"
echo "Installing other dependencies"
echo "========================================"
echo
pip install -r requirements.txt

# Check FFmpeg
echo
echo "========================================"
echo "Checking FFmpeg"
echo "========================================"
echo
if ! command -v ffmpeg &> /dev/null; then
    echo "WARNING: FFmpeg is not installed"
    echo
    echo "FFmpeg is required for audio conversion."
    echo
    echo "To install FFmpeg:"
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  Fedora:        sudo dnf install ffmpeg"
    echo "  macOS:         brew install ffmpeg"
    echo
else
    echo "FFmpeg found:"
    ffmpeg -version | head -n 1
fi

echo
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo
echo "To use the tools:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo
echo "  2. Run the music processor:"
echo "     python music_processor.py [YouTube URL]"
echo
echo "  Or use individual scripts:"
echo "     python youtube_downloader.py [URL]"
echo "     python stem_separator.py [audio_file]"
echo
echo "See README.md for detailed usage instructions."
echo
