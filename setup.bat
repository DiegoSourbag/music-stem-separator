@echo off
REM Setup script for Music Stem Separator (Windows)
REM Creates virtual environment and installs dependencies

echo ========================================
echo Music Stem Separator Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check if pip is available
pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: pip is not installed
    echo Please install pip
    pause
    exit /b 1
)

echo pip found:
pip --version
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist venv (
    echo Virtual environment already exists.
    choice /C YN /M "Do you want to recreate it"
    if errorlevel 2 goto :install
    echo Removing old virtual environment...
    rmdir /s /q venv
)

python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Virtual environment created successfully.
echo.

:install
REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA support (if available)
echo.
echo ========================================
echo Installing PyTorch
echo ========================================
echo.
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo NVIDIA GPU detected!
    echo Installing PyTorch with CUDA support...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo No NVIDIA GPU detected.
    echo Installing PyTorch (CPU-only version)...
    pip install torch torchvision torchaudio
)

REM Install other requirements
echo.
echo ========================================
echo Installing other dependencies
echo ========================================
echo.
pip install -r requirements.txt

REM Install FFmpeg (requires ffmpeg to be in PATH)
echo.
echo ========================================
echo Checking FFmpeg
echo ========================================
echo.
ffmpeg -version >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: FFmpeg is not installed or not in PATH
    echo.
    echo FFmpeg is required for audio conversion.
    echo Please install FFmpeg from: https://ffmpeg.org/download.html
    echo.
    echo After installation, add FFmpeg to your PATH environment variable.
    echo.
) else (
    echo FFmpeg found:
    ffmpeg -version | findstr "ffmpeg version"
)

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To use the tools:
echo   1. Activate the virtual environment:
echo      venv\Scripts\activate
echo.
echo   2. Run the music processor:
echo      python music_processor.py [YouTube URL]
echo.
echo   Or use individual scripts:
echo      python youtube_downloader.py [URL]
echo      python stem_separator.py [audio_file]
echo.
echo See README.md for detailed usage instructions.
echo.

pause
