# How to Run Music Stem Separator

Complete guide for running the application with GPU acceleration in terminal/command line.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Initial Setup](#initial-setup)
- [Running with GPU (Recommended)](#running-with-gpu-recommended)
- [Command Examples](#command-examples)
- [Web Interface](#web-interface)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required

1. **Python 3.8+**
   - Windows: Download from [python.org](https://www.python.org/)
   - Linux: Usually pre-installed, or `sudo apt install python3`
   - Mac: `brew install python3`

2. **FFmpeg**
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/) and add to PATH
   - Linux: `sudo apt install ffmpeg`
   - Mac: `brew install ffmpeg`

3. **NVIDIA GPU + CUDA** (for GPU acceleration)
   - NVIDIA GPU with CUDA support (GTX 1050 Ti or better recommended)
   - CUDA toolkit will be installed automatically with PyTorch

### Recommended Specs for GPU

- **Minimum**: GTX 1050 Ti, 4GB VRAM
- **Recommended**: RTX 2060 or higher, 6GB+ VRAM
- **Optimal**: RTX 3060 or higher, 8GB+ VRAM

---

## Initial Setup

### Step 1: Clone/Download the Repository

```bash
# If using git
git clone <repository-url>
cd music-stem-separator

# Or download and extract the ZIP file
```

### Step 2: Run Setup Script

This will create a virtual environment and install all dependencies with GPU support.

**Windows:**
```cmd
setup.bat
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
1. Create a Python virtual environment
2. Detect if you have an NVIDIA GPU (`nvidia-smi`)
3. Install PyTorch with CUDA 11.8 support (if GPU detected)
4. Install all other dependencies
5. Check for FFmpeg

### Step 3: Verify GPU Setup

```bash
# Windows
venv\Scripts\activate
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Linux/Mac
source venv/bin/activate
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Expected Output (with GPU):**
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 2060
```

**If CUDA is False:**
- Make sure you have an NVIDIA GPU
- Install/update NVIDIA drivers
- Re-run `setup.bat` or `setup.sh`

---

## Running with GPU (Recommended)

### Every Time You Want to Use the Application

**Step 1: Activate Virtual Environment**

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

You should see `(venv)` appear in your terminal prompt.

**Step 2: Run Your Desired Script**

The application will **automatically use GPU** if available. No special flags needed!

---

## Command Examples

### Basic Usage (Auto-uses GPU)

#### 1. Download + Separate All Stems

```bash
python music_processor.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

This will:
- Download audio from YouTube
- Separate into 4 stems (vocals, drums, bass, other)
- Use GPU automatically
- Save to `separated/htdemucs_ft/Song Name/`

#### 2. Create Karaoke Version (Fastest)

```bash
python karaoke.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

Output:
- `karaoke/Song Name/instrumental.mp3` (backing track)
- `karaoke/Song Name/vocals.mp3` (isolated vocals)

#### 3. High-Performance Mode (2-3x Faster with GPU)

```bash
# Karaoke with GPU boost
python karaoke.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --high-performance

# Full separation with GPU boost
python music_processor.py "URL" --high-performance
```

**Important**: High-performance mode requires ~3-4GB VRAM. Use only if you have RTX 2060 or better.

#### 4. Process Local File

```bash
python stem_separator.py "path/to/your/song.mp3"
```

#### 5. Batch Processing Multiple URLs

Create a text file `urls.txt`:
```
https://www.youtube.com/watch?v=VIDEO_ID_1
https://www.youtube.com/watch?v=VIDEO_ID_2
https://www.youtube.com/watch?v=VIDEO_ID_3
```

Run:
```bash
# Karaoke versions of all
python batch_processor.py urls.txt

# Full separation of all (with GPU boost)
python batch_processor.py urls.txt --mode full --high-performance
```

### Advanced Options

#### 6-Stem Separation (Separates Guitar and Piano)

```bash
python music_processor.py "URL" --model htdemucs_6s
```

Output: vocals, drums, bass, guitar, piano, other

#### Save as WAV Instead of MP3

```bash
python music_processor.py "URL" --stem-format wav
```

#### Use Specific Model

```bash
# Fast model (good quality, quick)
python music_processor.py "URL" --model mdx_extra

# Balanced (default, recommended)
python music_processor.py "URL" --model htdemucs_ft

# Best quality
python music_processor.py "URL" --model htdemucs

# Most detailed (6 stems)
python music_processor.py "URL" --model htdemucs_6s
```

#### Force CPU (if GPU has issues)

```bash
python music_processor.py "URL" --device cpu
```

---

## Web Interface

### Running Web UI with GPU

**Start the server:**

```bash
# Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Run the web app
python app.py
```

**Access:**
- Open browser to `http://localhost:5001`
- The web interface will use GPU automatically
- Can enable high-performance mode in the UI

### Web UI vs Command Line

| Feature | Web UI | Command Line |
|---------|--------|--------------|
| GPU Support | ✅ Yes (when run locally) | ✅ Yes |
| Ease of Use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Speed | Same | Same |
| Batch Processing | Limited | Full featured |
| Customization | Basic | Advanced |

---

## Performance Expectations

### With GPU (NVIDIA RTX 2060)

| Mode | Model | 5-min Song | Description |
|------|-------|------------|-------------|
| Standard | htdemucs_ft | ~1-1.5 min | Default, safe |
| High-Perf | htdemucs_ft | ~30-45 sec | 2-3x faster |
| Fast | mdx_extra | ~30 sec | Quick, good quality |
| Detailed | htdemucs_6s | ~2-3 min | 6 stems |
| Detailed+HP | htdemucs_6s | ~1 min | 6 stems, fast |

### With CPU (Modern i7)

| Model | 5-min Song |
|-------|------------|
| mdx_extra | ~3-5 min |
| htdemucs_ft | ~5-10 min |
| htdemucs | ~10-15 min |
| htdemucs_6s | ~15-25 min |

**Note**: High-performance mode has no effect on CPU.

---

## Troubleshooting

### GPU Not Being Used

**Check 1: Verify CUDA**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`:
1. Update NVIDIA drivers
2. Reinstall PyTorch with CUDA:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

**Check 2: Monitor GPU Usage**

While processing, open another terminal:
```bash
# Windows/Linux
nvidia-smi

# Should show python process using GPU
```

### Out of Memory Error

```
CUDA out of memory
```

**Solutions**:
1. Don't use high-performance mode:
   ```bash
   python music_processor.py "URL"  # Without --high-performance
   ```

2. Use CPU instead:
   ```bash
   python music_processor.py "URL" --device cpu
   ```

3. Use faster model:
   ```bash
   python music_processor.py "URL" --model mdx_extra
   ```

4. Close other GPU applications

### FFmpeg Not Found

```
ffmpeg: command not found
```

**Windows**:
1. Download FFmpeg from https://ffmpeg.org/
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to PATH environment variable
4. Restart terminal

**Linux**:
```bash
sudo apt update
sudo apt install ffmpeg
```

**Mac**:
```bash
brew install ffmpeg
```

Verify:
```bash
ffmpeg -version
```

### YouTube Download Fails

```
ERROR: unable to download video
```

**Solution**:
```bash
pip install --upgrade yt-dlp
```

### Virtual Environment Not Activating

**Windows**: If `venv\Scripts\activate` doesn't work:
```cmd
# Use activate.bat instead
venv\Scripts\activate.bat

# Or use PowerShell
venv\Scripts\Activate.ps1
```

**Linux/Mac**: Make sure you include `source`:
```bash
source venv/bin/activate  # Correct
venv/bin/activate         # Wrong - won't work
```

---

## Complete Workflow Example

Here's a complete example from start to finish:

```bash
# 1. Open terminal in project folder

# 2. Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 3. Verify GPU is available
python -c "import torch; print('GPU Ready!' if torch.cuda.is_available() else 'CPU Only')"

# 4. Process a song with GPU boost
python karaoke.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --high-performance

# 5. Check output
# Windows:
dir karaoke

# Linux/Mac:
ls karaoke/

# 6. Process another with full separation
python music_processor.py "https://www.youtube.com/watch?v=ANOTHER_ID" --high-performance

# 7. Done! Deactivate when finished
deactivate
```

---

## Quick Reference

### Most Common Commands

```bash
# Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Karaoke (fastest, most common use)
python karaoke.py "URL" --high-performance

# Full separation (all stems)
python music_processor.py "URL" --high-performance

# Batch process
python batch_processor.py urls.txt --high-performance

# Process local file
python stem_separator.py song.mp3 --high-performance

# Web interface
python app.py
```

### When to Use High-Performance Mode

**Use it (`--high-performance`) when:**
- You have RTX 2060 or better (6GB+ VRAM)
- Processing songs under 10 minutes
- Want maximum speed

**Don't use it when:**
- GPU has less than 4GB VRAM
- Processing very long files (>15 min)
- Getting "out of memory" errors
- Other GPU apps are running

---

## Getting Help

### Check Available Options

```bash
python music_processor.py --help
python karaoke.py --help
python batch_processor.py --help
python stem_separator.py --help
```

### List Available Models

```bash
python stem_separator.py --list-models
```

### View Smart Processor Modes

```bash
python smart_processor.py --list-modes
```

---

## Summary

**For fastest GPU processing:**
1. Activate virtual environment: `venv\Scripts\activate`
2. Run with high-performance: `python karaoke.py "URL" --high-performance`
3. Models download once and cache permanently
4. Subsequent runs are much faster

**For batch processing:**
1. Create `urls.txt` with one URL per line
2. Run: `python batch_processor.py urls.txt --high-performance`

**For web interface:**
1. Run: `python app.py`
2. Open: `http://localhost:5001`

That's it! The application handles GPU detection and usage automatically. Enjoy fast, high-quality stem separation!
