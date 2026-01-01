# Music Stem Separator

A comprehensive Python toolkit for downloading audio from YouTube and separating music into individual stems (vocals, drums, bass, instruments, etc.) using state-of-the-art AI models.

## Features

- **YouTube Audio Downloader**: Download high-quality audio from YouTube videos in MP3 or WAV format
- **AI-Powered Stem Separation**: Separate music into individual stems using Demucs models
- **Karaoke Creator**: Easily create karaoke/instrumental versions by removing vocals
- **Batch Processing**: Process multiple songs from a URL list file
- **GPU Acceleration**: Automatic NVIDIA GPU detection with CUDA support for faster processing
- **CPU Fallback**: Works on CPU-only systems (slower but functional)
- **Multiple Models**: Choose from different Demucs models based on your needs
- **Flexible Output**: Save stems as MP3 or WAV with customizable quality
- **Web Interface**: Browser-based UI for easy access (via Flask)
- **Docker Support**: Run in containers for easy deployment
- **Easy to Use**: Multiple interfaces - command-line, Python API, or web browser

## Supported Stem Types

Depending on the model used, you can separate music into:

- **4-stem models** (htdemucs, htdemucs_ft, mdx_extra):
  - Vocals
  - Drums
  - Bass
  - Other (all other instruments)

- **6-stem model** (htdemucs_6s):
  - Vocals
  - Drums
  - Bass
  - Guitar
  - Piano
  - Other

## Requirements

- Python 3.8 or higher
- pip (Python package manager)
- FFmpeg (for audio conversion)
- NVIDIA GPU with CUDA support (optional, for faster processing)

## Installation

### Windows

1. Clone or download this repository
2. Open Command Prompt in the project folder
3. Run the setup script:
   ```cmd
   setup.bat
   ```

### Linux/Mac

1. Clone or download this repository
2. Open terminal in the project folder
3. Run the setup script:
   ```bash
   ./setup.sh
   ```

### Manual Installation

If you prefer to set up manually:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. Install dependencies:
   ```bash
   # For GPU support (NVIDIA CUDA)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # For CPU-only
   pip install torch torchvision torchaudio

   # Install other requirements
   pip install -r requirements.txt
   ```

4. Install FFmpeg:
   - Windows: Download from https://ffmpeg.org/download.html and add to PATH
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`

### Docker (Web UI)

If you have Docker and Docker Compose installed, you can run the web interface in a container.

**Note**: Docker version uses CPU-only. For GPU acceleration, use the local Python installation below.

1.  **Build the image:**
    ```bash
    docker-compose build
    ```

2.  **Run the container:**
    ```bash
    docker-compose up -d
    ```

3.  **Access the web interface:**
    Open your browser and go to `http://127.0.0.1:5001`.

To view the logs:
```bash
docker-compose logs -f
```

To stop the container:
```bash
docker-compose down
```

## Usage

**Choose Your Interface:**
- **Command Line (GPU Recommended)**: Fast, full GPU support, detailed below
- **Web Interface**: Browser-based, easier to use, see [Web Interface](#web-interface) section
- **Python API**: For integration, see [Python API Usage](#python-api-usage) section

### Quick Start: All-in-One Processor (Command Line)

The easiest way to download and separate music:

```bash
# Activate virtual environment first
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Download from YouTube and separate into all stems
python music_processor.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Extract only vocals (creates vocals and no_vocals)
python music_processor.py "URL" --two-stems vocals

# Use 6-stem model for detailed separation
python music_processor.py "URL" --model htdemucs_6s

# Use CPU instead of GPU
python music_processor.py "URL" --device cpu

# Save stems as WAV files
python music_processor.py "URL" --stem-format wav
```

### Individual Scripts

#### 1. YouTube Downloader

Download audio from YouTube:

```bash
# Download as MP3 (default)
python youtube_downloader.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Download as WAV
python youtube_downloader.py "URL" --format wav

# Custom filename and quality
python youtube_downloader.py "URL" --output my_song --quality 192

# Custom output directory
python youtube_downloader.py "URL" --output-dir my_downloads
```

#### 2. Stem Separator

Separate an existing audio file:

```bash
# Separate into all stems
python stem_separator.py song.mp3

# Use 6-stem model
python stem_separator.py song.mp3 --model htdemucs_6s

# Extract only vocals
python stem_separator.py song.mp3 --two-stems vocals

# Save as WAV
python stem_separator.py song.mp3 --wav

# Use CPU instead of GPU
python stem_separator.py song.mp3 --device cpu

# List available models
python stem_separator.py --list-models
```

#### 3. Karaoke Creator

Create karaoke/instrumental versions by removing vocals:

```bash
# Create karaoke from YouTube
python karaoke.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Process local file
python karaoke.py --local song.mp3

# Save as WAV format
python karaoke.py "URL" --format wav

# Don't save vocals track (only instrumental)
python karaoke.py "URL" --no-vocals

# Use 6-stem model for better quality
python karaoke.py "URL" --model htdemucs_6s

# Enable high-performance GPU mode (2-3x faster, uses more VRAM)
python karaoke.py "URL" --high-performance
```

Output files saved to `karaoke/Song Name/`:
- `instrumental.mp3` - Karaoke/backing track (no vocals) - **automatically combines all non-vocal stems**
- `vocals.mp3` - Isolated vocals (optional)

#### 4. Smart Processor (Recommended)

Intelligent processor with automatic model selection and time estimation:

```bash
# Auto mode with time estimate and confirmation
python smart_processor.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Skip confirmation prompt
python smart_processor.py "URL" --yes

# Auto-select for guitar separation
python smart_processor.py "URL" --need-guitar --mode full

# Fast mode for quick results
python smart_processor.py "URL" --quality fast

# See all available quality modes
python smart_processor.py --list-modes
```

**Features:**
- Shows estimated processing time before starting
- Asks for confirmation (can skip with `--yes`)
- Auto-selects best model based on requirements
- Adapts to CPU vs GPU hardware

#### 5. Batch Processor

Process multiple songs from a URL list file:

```bash
# Create karaoke versions for all URLs
python batch_processor.py urls.txt

# Full stem separation for all URLs
python batch_processor.py urls.txt --mode full

# Use 6-stem model and WAV format
python batch_processor.py urls.txt --model htdemucs_6s --format wav

# Add 5-second delay between downloads
python batch_processor.py urls.txt --delay 5

# Save error log file
python batch_processor.py urls.txt --save-errors

# Enable high-performance GPU mode for faster processing
python batch_processor.py urls.txt --high-performance
```

**URL List File Format** (see `urls.example.txt`):
```
# One URL per line, comments start with #
https://www.youtube.com/watch?v=VIDEO_ID_1
https://www.youtube.com/watch?v=VIDEO_ID_2

# Comments and empty lines are ignored
https://youtu.be/SHORT_ID
```

## Available Models

| Model | Stems | Quality | Speed | Best For |
|-------|-------|---------|-------|----------|
| `htdemucs` | 4 | Excellent | Slow | Highest quality |
| `htdemucs_ft` | 4 | Excellent | Medium | **Best overall (default)** |
| `htdemucs_6s` | 6 | Excellent | Slow | Detailed instrument separation |
| `mdx_extra` | 4 | Good | Fast | Quick processing |

## Performance Optimization

### High-Performance GPU Mode

All separation scripts now support `--high-performance` mode for maximum GPU utilization:

**Standard Mode** (default):
- Uses ~5-15% GPU
- Splits audio into chunks to save VRAM
- Safer for systems with limited VRAM
- Slower processing

**High-Performance Mode** (`--high-performance`):
- Uses ~80-100% GPU
- Loads entire song into GPU memory
- 2-3x faster processing
- Requires ~3-4GB VRAM for typical songs
- **Recommended for RTX 2060 and higher**

**Example:**
```bash
# Standard mode (conservative)
python karaoke.py "URL"

# High-performance mode (2-3x faster)
python karaoke.py "URL" --high-performance

# Batch processing with high performance
python batch_processor.py urls.txt --high-performance
```

**When to use high-performance mode:**
- You have 6GB+ VRAM (RTX 2060, RTX 3060, etc.)
- Processing songs under 10 minutes
- You want maximum speed

**When NOT to use:**
- Limited VRAM (< 4GB)
- Very long audio files (> 15 minutes)
- Running other GPU applications simultaneously

## Command-Line Options

### music_processor.py (Coordinator)

```
usage: music_processor.py [-h] [--filename FILENAME]
                          [--download-format {mp3,wav}]
                          [--download-dir DOWNLOAD_DIR]
                          [--model {htdemucs,htdemucs_ft,htdemucs_6s,mdx_extra}]
                          [--device {cuda,cpu}]
                          [--output-dir OUTPUT_DIR]
                          [--two-stems {vocals,drums,bass,other,guitar,piano}]
                          [--stem-format {mp3,wav}]
                          [--mp3-bitrate MP3_BITRATE]
                          [--no-keep-original]
                          [--list-models]
                          [--local FILE | url]

Options:
  url                   YouTube URL to download and process
  --local FILE, -l FILE Process local audio file instead
  --filename FILENAME   Custom filename for download
  --download-format     Format for downloaded audio (mp3/wav)
  --download-dir        Directory for downloads
  --model, -m           Demucs model to use
  --device, -d          Device (cuda/cpu)
  --output-dir, -o      Directory for separated stems
  --two-stems           Extract only two stems
  --stem-format         Output format for stems (mp3/wav)
  --mp3-bitrate         MP3 bitrate in kbps
  --no-keep-original    Don't keep original download
  --list-models         List available models
```

### youtube_downloader.py

```
Options:
  url                   YouTube URL to download
  --output, -o          Output filename (without extension)
  --format, -f          Audio format (mp3/wav)
  --quality, -q         Audio quality/bitrate for MP3
  --output-dir, -d      Output directory
```

### stem_separator.py

```
Options:
  audio_file            Audio file to separate
  --model, -m           Model to use
  --device, -d          Device (cuda/cpu)
  --output-dir, -o      Output directory
  --two-stems           Extract only two stems
  --wav                 Save as WAV instead of MP3
  --mp3-bitrate         MP3 bitrate in kbps
  --float32             Save as 32-bit float WAV
  --int24               Save as 24-bit int WAV
  --high-performance    Enable high-performance GPU mode (faster)
  --list-models         List available models
```

## Output Structure

```
music-stem-separator/
├── downloads/              # Downloaded audio files
│   └── song_name.mp3
└── separated/              # Separated stems
    └── htdemucs_ft/       # Model name
        └── song_name/     # Song folder
            ├── vocals.mp3
            ├── drums.mp3
            ├── bass.mp3
            └── other.mp3
```

## Performance Tips

### GPU vs CPU

- **GPU (NVIDIA CUDA)**:
  - ~5-10x faster than CPU
  - Recommended for frequent use
  - Requires NVIDIA GPU with CUDA support
  - Automatically detected by the scripts

- **CPU**:
  - Works on any system
  - Slower but produces identical results
  - Good for occasional use
  - Use `--device cpu` to force CPU mode

### Model Selection

- **For best quality**: Use `htdemucs` or `htdemucs_6s`
- **For balanced performance**: Use `htdemucs_ft` (default)
- **For speed**: Use `mdx_extra`

### Processing Time Estimates

For a 4-minute song:
- **GPU (RTX 3060)**: 30-90 seconds
- **CPU (modern i7)**: 5-15 minutes

## Troubleshooting

### FFmpeg Not Found

**Error**: `ffmpeg: command not found`

**Solution**:
- Windows: Download FFmpeg from https://ffmpeg.org/download.html and add to PATH
- Linux: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`

### CUDA Out of Memory

**Error**: `CUDA out of memory`

**Solution**:
- Use CPU mode: `--device cpu`
- Process shorter audio files
- Close other GPU-intensive applications

### YouTube Download Fails

**Error**: Download errors or HTTP 403

**Solution**:
- Update yt-dlp: `pip install --upgrade yt-dlp`
- Try a different URL
- Check your internet connection

### Poor Separation Quality

**Solution**:
- Try a different model (htdemucs or htdemucs_6s)
- Ensure input audio is high quality
- Some songs separate better than others due to mixing

## Web Interface

The application includes a Flask-based web interface for easy browser access.

### Starting the Web Server

**Option 1: Docker (CPU-only)**
```bash
docker-compose up -d
# Access at http://localhost:5001
```

**Option 2: Local Python (with GPU support)**
```bash
# Activate virtual environment first
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Start the web server
python app.py

# Access at http://localhost:5001
```

### Web UI Features

- **URL Input**: Paste single or multiple YouTube URLs
- **Processing Modes**:
  - Karaoke (instrumental + vocals)
  - 4-Stem Separation
  - 6-Stem Separation
  - Download Only
- **Model Selection**: Choose quality vs speed
- **High-Performance Mode**: Enable GPU boost (2-3x faster)
- **Progress Tracking**: Real-time progress updates
- **File Downloads**: Direct download links for output files

## Python API Usage

You can also use the scripts as Python modules:

```python
from youtube_downloader import YouTubeAudioDownloader
from stem_separator import StemSeparator
from music_processor import MusicProcessor
from karaoke import KaraokeCreator

# Download audio
downloader = YouTubeAudioDownloader(output_dir='downloads', format='mp3')
audio_file = downloader.download('https://www.youtube.com/watch?v=VIDEO_ID')

# Separate stems (with GPU)
separator = StemSeparator(model_name='htdemucs_ft', device='cuda', high_performance=True)
stems = separator.separate(audio_file)

# Create karaoke version
karaoke = KaraokeCreator(model='htdemucs_ft', device='cuda', high_performance=True)
result = karaoke.create_from_youtube('URL')

# Or use the all-in-one processor
processor = MusicProcessor(model='htdemucs_ft', device='cuda', high_performance=True)
audio_file, stems = processor.process_from_youtube('URL')
```

## Project Structure

```
music-stem-separator/
├── youtube_downloader.py   # YouTube audio downloader
├── stem_separator.py       # Stem separation engine
├── music_processor.py      # Coordinator script
├── karaoke.py             # Karaoke/instrumental creator
├── batch_processor.py     # Batch URL processor
├── separate_simple.py     # Standalone separator with fixes
├── example.py             # API usage examples
├── requirements.txt       # Python dependencies
├── setup.bat              # Windows setup script
├── setup.sh               # Linux/Mac setup script
├── README.md              # This file
├── QUICKSTART.md          # Quick start guide
├── CLAUDE.md              # Claude Code instructions
├── urls.example.txt       # Example URL list file
├── config.example.py      # Configuration template
├── downloads/             # Downloaded audio (created at runtime)
├── separated/             # Separated stems (created at runtime)
└── karaoke/               # Karaoke versions (created at runtime)
```

## Technical Details

### Dependencies

- **yt-dlp**: Modern YouTube downloader
- **Demucs**: State-of-the-art music source separation
- **PyTorch**: Deep learning framework
- **FFmpeg**: Audio/video processing
- **torchaudio**: Audio I/O for PyTorch

### Models

The project uses Demucs models trained on large datasets of music:
- Trained on hundreds of hours of music
- Uses deep neural networks (hybrid CNN-Transformer architecture)
- Achieves state-of-the-art separation quality

### GPU Acceleration

- Uses CUDA for NVIDIA GPUs
- Automatically falls back to CPU if GPU unavailable
- Supports mixed precision for faster processing

## License

This project is provided as-is for educational and personal use. Please respect copyright laws when downloading and processing music.

**Note**: Only download and process music you have the rights to use.

## Credits

- **Demucs**: https://github.com/facebookresearch/demucs
- **yt-dlp**: https://github.com/yt-dlp/yt-dlp
- **PyTorch**: https://pytorch.org/

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Ensure all dependencies are correctly installed
3. Verify FFmpeg is in your PATH
4. Try updating yt-dlp: `pip install --upgrade yt-dlp`

## Version

Current Version: 1.0.0

---

Happy music processing! 🎵
