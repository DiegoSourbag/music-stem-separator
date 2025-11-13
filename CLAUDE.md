# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This repository provides a Python toolkit for downloading audio from YouTube and separating music into individual stems (vocals, drums, bass, instruments) using Facebook's Demucs AI models. It supports GPU acceleration with NVIDIA CUDA and CPU fallback.

## Environment Setup

### Windows Environment
- Python command is `python.exe` (not `python` or `python3`)
- Virtual environment activation: `venv\Scripts\activate.bat`
- Setup script: `setup.bat` (automatically detects NVIDIA GPU via `nvidia-smi`)

### Dependencies Installation
For GPU support (NVIDIA CUDA 11.8):
```bash
python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python.exe -m pip install -r requirements.txt
```

For CPU-only:
```bash
python.exe -m pip install torch torchvision torchaudio
python.exe -m pip install -r requirements.txt
```

### FFmpeg Requirement
FFmpeg must be installed and in PATH for audio conversion. The scripts will fail without it.

## Architecture Overview

### Core Architecture
1. **YouTubeAudioDownloader** (`youtube_downloader.py`) - Downloads audio using yt-dlp
2. **StemSeparator** (`stem_separator.py`) - Separates stems using Demucs models
3. **MusicProcessor** (`music_processor.py`) - Coordinator that chains download → separation
4. **KaraokeCreator** (`karaoke.py`) - Specialized tool for creating karaoke/instrumental versions
5. **BatchProcessor** (`batch_processor.py`) - Processes multiple URLs from a text file

Each component can be used independently or through higher-level coordinators.

### Key Components

#### YouTubeAudioDownloader
- Uses yt-dlp with FFmpeg post-processing
- Sanitizes filenames by removing non-alphanumeric characters (except spaces, hyphens, underscores)
- Default output: `downloads/` directory
- Formats: MP3 (default, 320kbps) or WAV

#### StemSeparator
- Wraps Facebook's Demucs models via `demucs.pretrained.get_model()`
- Device detection: Auto-detects CUDA GPU, falls back to CPU
- Model lazy-loading: Downloads on first use (~80MB per model file)
- Processing pipeline: AudioFile → numpy → normalization → Tensor → apply_model → save stems
- Default output: `separated/<model_name>/<song_name>/` directory
- **High-performance mode**: `high_performance=True` parameter enables full GPU utilization (2-3x faster)

**Critical Technical Detail**: The `AudioFile.read()` method can return either a numpy array or a Tensor depending on the Demucs version. Always check the type and convert if needed:
```python
wav = AudioFile(str(audio_path)).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
if isinstance(wav, torch.Tensor):
    wav = wav.numpy()
```

**GPU Performance Modes**:
- **Standard mode** (default): `split=True` - processes audio in chunks, uses ~5-15% GPU, safer for limited VRAM
- **High-performance mode**: `split=False` - loads entire song into GPU memory, uses ~80-100% GPU, requires ~3-4GB VRAM for typical songs, 2-3x faster

#### KaraokeCreator
- Specialized tool for creating karaoke/instrumental tracks
- Uses `two_stems='vocals'` mode internally to split audio into vocals and everything else
- **Key behavior**: Automatically combines all non-vocal stems (bass, drums, other) into a single `instrumental.mp3` file
- Output: `karaoke/Song Name/instrumental.mp3` (combined backing track) + optional `vocals.mp3`
- User does NOT need to manually combine stems - Demucs does this automatically via the two-stems mode

#### MusicProcessor
- Chains downloader and separator
- Manages cleanup of intermediate files via `--no-keep-original` flag
- Provides unified command-line interface
- Passes through `high_performance` parameter to StemSeparator

### Demucs Models

| Model | Stems | Speed | Use Case |
|-------|-------|-------|----------|
| `htdemucs` | 4 | Slow | Highest quality |
| `htdemucs_ft` | 4 | Medium | **Default - best balance** |
| `htdemucs_6s` | 6 | Slow | Detailed instruments (adds guitar, piano) |
| `mdx_extra` | 4 | Fast | Quick processing |

4-stem models produce: `vocals`, `drums`, `bass`, `other`
6-stem model adds: `guitar`, `piano` (separates from `other`)

### Processing Flow

```
YouTube URL → download (yt-dlp + FFmpeg) → MP3/WAV file
  ↓
Load audio → resample to model.samplerate → normalize
  ↓
Convert to Tensor → move to device (CUDA/CPU)
  ↓
apply_model() with shifts=1, split=True, overlap=0.25, progress=True
  ↓
Save individual stems (MP3 320kbps or WAV)
```

## Common Commands

### Setup and Activation
```bash
# Initial setup (Windows)
setup.bat

# Activate environment (required before running scripts)
venv\Scripts\activate.bat

# Verify GPU detection
python.exe -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Running Scripts

**All-in-one processing:**
```bash
python.exe music_processor.py "https://youtube.com/watch?v=VIDEO_ID"
python.exe music_processor.py "URL" --two-stems vocals
python.exe music_processor.py "URL" --model htdemucs_6s
python.exe music_processor.py --local song.mp3
```

**Download only:**
```bash
python.exe youtube_downloader.py "URL"
python.exe youtube_downloader.py "URL" --format wav --quality 192
```

**Separate only:**
```bash
python.exe stem_separator.py song.mp3
python.exe stem_separator.py song.mp3 --model htdemucs_6s --device cpu
python.exe stem_separator.py song.mp3 --two-stems vocals --wav
```

**Create karaoke versions:**
```bash
python.exe karaoke.py "https://youtube.com/watch?v=VIDEO_ID"
python.exe karaoke.py --local song.mp3
python.exe karaoke.py "URL" --format wav --no-vocals
python.exe karaoke.py "URL" --high-performance  # 2-3x faster with full GPU
```

**Batch processing:**
```bash
python.exe batch_processor.py urls.txt
python.exe batch_processor.py urls.txt --mode full
python.exe batch_processor.py urls.txt --delay 5 --save-errors
python.exe batch_processor.py urls.txt --high-performance  # Maximum GPU utilization
```

**List available models:**
```bash
python.exe stem_separator.py --list-models
```

## Common Issues and Solutions

### Unicode/Console Encoding (Windows)
Windows console may not support Unicode characters (✓, ✗). For scripts with console output, wrap stdout/stderr:
```python
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
```

### Type Errors with AudioFile
If seeing "expected np.ndarray (got Tensor)", the AudioFile.read() is returning a Tensor. Add type check:
```python
wav = AudioFile(str(audio_path)).read(...)
if isinstance(wav, torch.Tensor):
    wav = wav.numpy()
```

### CUDA Out of Memory
- Disable high-performance mode (remove `--high-performance` flag)
- Use `--device cpu` flag
- Process shorter audio files
- Close other GPU applications
- Consider using `mdx_extra` model (lighter)
- High-performance mode requires ~3-4GB VRAM; standard mode uses less memory

### yt-dlp Download Failures
- Update: `pip install --upgrade yt-dlp`
- Check internet connection
- Some videos may be region-locked or have restrictions

### FFmpeg Not Found
- Windows: Download from ffmpeg.org and add bin directory to PATH
- Verify with: `ffmpeg -version`

## Output Structure

```
music-stem-separator/
├── downloads/                    # Downloaded audio files
│   └── Song Name.mp3
├── separated/                    # Separated stems
│   └── htdemucs_ft/             # Model-specific folder
│       └── Song Name/           # Song-specific folder
│           ├── vocals.mp3       # Individual stems
│           ├── drums.mp3
│           ├── bass.mp3
│           └── other.mp3
├── karaoke/                      # Karaoke/instrumental versions
│   └── Song Name/
│       ├── instrumental.mp3     # Backing track (no vocals)
│       └── vocals.mp3           # Isolated vocals (optional)
├── youtube_downloader.py         # Core: Download
├── stem_separator.py            # Core: Separation
├── music_processor.py           # Coordinator: Download + Separate
├── karaoke.py                   # Specialized: Karaoke creator
├── batch_processor.py           # Batch: URL list processor
├── separate_simple.py           # Standalone separator with encoding fixes
├── example.py                   # Interactive API usage examples
└── urls.example.txt             # Example URL list file
```

## Python API Usage

```python
from youtube_downloader import YouTubeAudioDownloader
from stem_separator import StemSeparator
from music_processor import MusicProcessor
from karaoke import KaraokeCreator

# Individual components
downloader = YouTubeAudioDownloader(output_dir='downloads', format='mp3')
audio_file = downloader.download('https://youtube.com/watch?v=ID')

# Standard separation
separator = StemSeparator(model_name='htdemucs_ft', device='cuda')
stems = separator.separate(audio_file, two_stems='vocals', mp3=True)

# High-performance separation (2-3x faster)
separator_fast = StemSeparator(model_name='htdemucs_ft', device='cuda', high_performance=True)
stems = separator_fast.separate(audio_file, mp3=True)

# Karaoke creation (automatically combines non-vocal stems)
karaoke = KaraokeCreator(model='htdemucs_ft', device='cuda', high_performance=True)
result = karaoke.create_from_youtube('URL')  # Returns {'instrumental': 'path/to/file.mp3', 'vocals': '...'}

# All-in-one processor
processor = MusicProcessor(model='htdemucs_ft', device='cuda', high_performance=True)
audio_file, stems = processor.process_from_youtube('URL', keep_original=True)
```

## Performance Expectations

### GPU (NVIDIA RTX 2060 with Max-Q, 6.44 GB VRAM)
**Standard Mode:**
- Model: htdemucs_ft
- 6-minute song: ~3-4 minutes processing
- GPU utilization: ~5-15%
- 4 stems of ~14MB each

**High-Performance Mode (`--high-performance`):**
- Model: htdemucs_ft
- 6-minute song: ~1-2 minutes processing (2-3x faster)
- GPU utilization: ~80-100%
- Same output quality and file sizes
- Recommended for RTX 2060 and higher

### CPU (Modern i7)
- Model: htdemucs_ft
- 6-minute song: ~5-15 minutes processing
- Identical quality to GPU
- High-performance flag has no effect on CPU

## Testing During Development

When making changes to separation logic:
1. Use a short test file (30-60 seconds) to iterate quickly
2. Test with both GPU and CPU modes
3. Verify output file sizes are reasonable (similar to input size × number of stems)
4. Check that all expected stem files are created
5. Test error handling with invalid inputs (missing files, bad URLs)

## Important Notes

- The `separate_simple.py` script is a standalone version with Windows encoding fixes and proper tensor handling - use it as a reference for robust implementation
- Models are downloaded automatically to `~/.cache/torch/hub/` on first use and cached permanently
- Each model requires ~320MB of download (4 files × ~80MB)
- The coordinator (`music_processor.py`) initializes the separator lazily to avoid loading models unnecessarily
- Always activate the virtual environment before running scripts
- **High-performance mode is recommended for RTX 2060 and higher** - provides 2-3x speedup with full GPU utilization
- All scripts that perform separation accept the `--high-performance` flag: `stem_separator.py`, `karaoke.py`, `music_processor.py`, `batch_processor.py`
- The karaoke feature automatically creates a single combined instrumental track - no manual stem combination needed
