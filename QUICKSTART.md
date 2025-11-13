# Quick Start Guide

Get up and running with Music Stem Separator in 5 minutes!

## 1. Installation

### Windows
```cmd
setup.bat
```

### Linux/Mac
```bash
./setup.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Detect and configure GPU support (if available)

## 2. Activate Virtual Environment

### Windows
```cmd
venv\Scripts\activate
```

### Linux/Mac
```bash
source venv/bin/activate
```

## 3. Basic Usage

### Download and Separate Everything

```bash
python music_processor.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

This will:
1. Download the audio from YouTube
2. Separate it into 4 stems: vocals, drums, bass, other
3. Save files in the `separated/` folder

### Extract Only Vocals

```bash
python music_processor.py "URL" --two-stems vocals
```

Creates two files:
- `vocals.mp3` - Just the vocals
- `no_vocals.mp3` - Everything except vocals (instrumental)

### Separate a Local File

```bash
python stem_separator.py path/to/your/song.mp3
```

## 4. Output Location

Your files will be saved in:
```
separated/
  └── htdemucs_ft/           # Model name
      └── song_name/         # Your song
          ├── vocals.mp3
          ├── drums.mp3
          ├── bass.mp3
          └── other.mp3
```

## 5. Common Commands

### Download as WAV
```bash
python music_processor.py "URL" --stem-format wav
```

### Use CPU (no GPU)
```bash
python music_processor.py "URL" --device cpu
```

### 6-Stem Separation (more detailed)
```bash
python music_processor.py "URL" --model htdemucs_6s
```
Produces: vocals, drums, bass, guitar, piano, other

### Process Local File
```bash
python music_processor.py --local my_song.mp3
```

### Create Karaoke Version
```bash
# Create karaoke from YouTube
python karaoke.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Process local file
python karaoke.py --local my_song.mp3
```

### Process Multiple Songs
```bash
# Create a text file with URLs (one per line)
# Then process all at once
python batch_processor.py urls.txt

# For full stem separation
python batch_processor.py urls.txt --mode full
```

## GPU vs CPU

- **With GPU**: ~1 minute per song
- **Without GPU**: ~5-15 minutes per song

Both produce identical quality results!

## Troubleshooting

### Can't find FFmpeg?
- Windows: Download from https://ffmpeg.org/ and add to PATH
- Linux: `sudo apt-get install ffmpeg`
- Mac: `brew install ffmpeg`

### YouTube download fails?
```bash
pip install --upgrade yt-dlp
```

### Out of memory?
```bash
python music_processor.py "URL" --device cpu
```

## Next Steps

- Read the full [README.md](README.md) for advanced options
- Try the [example.py](example.py) script for programmatic usage
- Experiment with different models using `--list-models`

## Help

For more help on any script:
```bash
python music_processor.py --help
python youtube_downloader.py --help
python stem_separator.py --help
```

---

Happy separating! 🎵
