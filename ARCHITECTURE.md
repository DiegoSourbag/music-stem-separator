# Architecture Documentation

Complete technical architecture documentation for the Music Stem Separator project.

## Table of Contents

- [System Overview](#system-overview)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Processing Pipeline](#processing-pipeline)
- [Web Interface](#web-interface)
- [Deployment Options](#deployment-options)
- [Technology Stack](#technology-stack)

## System Overview

The Music Stem Separator is a modular Python application that combines YouTube audio downloading with AI-powered stem separation using Facebook's Demucs models. The system supports both command-line and web-based interfaces.

### Key Features

- **Modular Architecture**: Independent components that can be used standalone or orchestrated
- **Multiple Interfaces**: CLI scripts, Python API, and Flask web UI
- **GPU Acceleration**: Automatic NVIDIA CUDA detection with CPU fallback
- **Flexible Processing**: Support for different models, quality modes, and output formats
- **Batch Processing**: Handle multiple URLs with error recovery and progress tracking

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interfaces                           │
├─────────────────┬──────────────────┬────────────────────────────┤
│  CLI Scripts    │  Python API      │  Web UI (Flask)            │
│  - Individual   │  - Direct import │  - Browser interface       │
│  - Batch        │  - Programmatic  │  - Background processing   │
│  - Smart        │    usage         │  - Progress tracking       │
└─────────────────┴──────────────────┴────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Coordinator Layer                             │
├─────────────────┬──────────────────┬────────────────────────────┤
│ MusicProcessor  │ SmartProcessor   │  BatchProcessor            │
│ (Orchestrator)  │ (Auto-select)    │  (Multi-URL)               │
└─────────────────┴──────────────────┴────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Core Services                               │
├──────────────────────────────┬──────────────────────────────────┤
│  YouTubeAudioDownloader      │  StemSeparator                   │
│  - yt-dlp wrapper            │  - Demucs wrapper                │
│  - Audio format conversion   │  - Model management              │
│  - Filename sanitization     │  - GPU/CPU processing            │
└──────────────────────────────┴──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    External Dependencies                         │
├──────────────────┬──────────────┬──────────────┬────────────────┤
│  yt-dlp          │  Demucs      │  PyTorch     │  FFmpeg        │
│  (YouTube DL)    │  (AI Models) │  (Deep Learn)│  (Audio Proc)  │
└──────────────────┴──────────────┴──────────────┴────────────────┘
```

## Core Components

### 1. YouTubeAudioDownloader (`youtube_downloader.py`)

**Purpose**: Downloads audio from YouTube videos and converts to desired format.

**Key Responsibilities**:
- URL validation and metadata extraction
- Audio download via yt-dlp
- Format conversion (MP3/WAV) via FFmpeg
- Filename sanitization (removes special characters)
- Quality/bitrate control

**API**:
```python
downloader = YouTubeAudioDownloader(
    output_dir='downloads',
    format='mp3',
    quality='320'
)
audio_file = downloader.download(url, filename=None)
```

**Output**: Path to downloaded audio file

**Dependencies**: yt-dlp, FFmpeg

---

### 2. StemSeparator (`stem_separator.py`)

**Purpose**: Separates audio into individual stems using Demucs AI models.

**Key Responsibilities**:
- Model loading and caching
- Device selection (CUDA/CPU)
- Audio preprocessing (normalization, resampling)
- Stem separation via Demucs
- Output format conversion (MP3/WAV)
- Performance mode selection (standard/high-performance)

**API**:
```python
separator = StemSeparator(
    model_name='htdemucs_ft',
    device='cuda',
    high_performance=False
)
stems = separator.separate(
    audio_file,
    output_dir='separated',
    two_stems=None,  # or 'vocals' for 2-stem mode
    mp3=True
)
```

**Output**: Dictionary mapping stem names to file paths

**Models Available**:
- `htdemucs`: Highest quality, 4 stems
- `htdemucs_ft`: Balanced quality/speed (default), 4 stems
- `htdemucs_6s`: Detailed separation, 6 stems (adds guitar, piano)
- `mdx_extra`: Fastest, 4 stems

**Performance Modes**:
- **Standard** (`split=True`): Processes audio in chunks, ~5-15% GPU usage
- **High-Performance** (`split=False`): Loads entire song, ~80-100% GPU usage, 2-3x faster

**Dependencies**: Demucs, PyTorch, torchaudio

---

### 3. MusicProcessor (`music_processor.py`)

**Purpose**: Orchestrates download + separation workflow.

**Key Responsibilities**:
- Coordinates YouTubeAudioDownloader and StemSeparator
- Handles file cleanup (optional)
- Provides unified command-line interface
- Supports both YouTube URLs and local files

**API**:
```python
processor = MusicProcessor(
    model='htdemucs_ft',
    device='cuda',
    high_performance=False
)

# From YouTube
audio_file, stems = processor.process_from_youtube(
    url,
    keep_original=True
)

# From local file
stems = processor.process_from_file(audio_file)
```

**Workflow**:
1. Download audio (if URL provided)
2. Initialize separator (lazy loading)
3. Separate stems
4. Optionally delete original download
5. Return paths to all files

---

### 4. KaraokeCreator (`karaoke.py`)

**Purpose**: Specialized processor for creating karaoke/instrumental tracks.

**Key Responsibilities**:
- Uses `two_stems='vocals'` mode internally
- Automatically combines non-vocal stems into single instrumental track
- Optional vocal track output
- Simplified workflow for karaoke use case

**API**:
```python
karaoke = KaraokeCreator(
    model='htdemucs_ft',
    device='cuda',
    high_performance=False
)

result = karaoke.create_from_youtube(
    url,
    keep_vocals=True,
    format='mp3'
)
# Returns: {'instrumental': 'path/to/file.mp3', 'vocals': 'path/to/file.mp3'}
```

**Output Structure**:
```
karaoke/
└── Song Name/
    ├── instrumental.mp3  # Combined backing track (no vocals)
    └── vocals.mp3        # Isolated vocals (optional)
```

---

### 5. SmartProcessor (`smart_processor.py`)

**Purpose**: Intelligent processor with automatic model selection and time estimation.

**Key Responsibilities**:
- Hardware detection (GPU/CPU)
- Automatic model selection based on requirements
- Processing time estimation
- User confirmation workflow
- Quality mode abstraction

**Quality Modes**:
- **fast**: `mdx_extra` - Quick processing, good quality
- **balanced**: `htdemucs_ft` - Best quality/speed (default)
- **quality**: `htdemucs` - Highest quality 4-stem
- **detailed**: `htdemucs_6s` - Full 6-stem separation

**Auto-Selection Logic**:
```python
if need_guitar or need_piano:
    → detailed mode (htdemucs_6s)
elif prefer_speed:
    → fast mode (mdx_extra)
elif device == 'cpu':
    → fast mode (mdx_extra)
else:
    → balanced mode (htdemucs_ft)
```

**Time Estimation**:
- Uses empirical factors based on model and device
- GPU factors: 0.125x (fast) to 0.5x (detailed)
- CPU factors: 0.75x (fast) to 3.0x (detailed)

---

### 6. BatchProcessor (`batch_processor.py`)

**Purpose**: Process multiple URLs from a text file.

**Key Responsibilities**:
- URL file parsing (supports comments, empty lines)
- Sequential processing with progress tracking
- Error handling and logging
- Optional delay between downloads (avoid rate limiting)
- Supports all processing modes (karaoke, full, download-only)

**URL File Format**:
```
# Lines starting with # are comments
https://www.youtube.com/watch?v=VIDEO_ID_1
https://youtu.be/SHORT_ID_2

# Empty lines are ignored
https://www.youtube.com/watch?v=VIDEO_ID_3
```

**Error Handling**:
- Continues on failure
- Logs errors to console and optional file
- Tracks success/failure count

---

### 7. Web UI (`app.py`)

**Purpose**: Flask-based web interface for browser-based interaction.

**Key Features**:
- Form-based URL input (single or multiple)
- Processing mode selection (karaoke, 4-stem, 6-stem, download-only)
- Model and performance options
- Background processing with threading
- Real-time progress tracking
- File download links

**Architecture**:
```python
# Job State Management
job_status = {
    'is_running': bool,
    'progress': int,
    'total': int,
    'current_task': str,
    'error': str | None,
    'output_files': List[str]
}

# Background Processing
def run_processing_job(urls, mode, model, high_performance):
    # Runs in separate thread
    # Updates job_status as it progresses
    pass

# Routes
@app.route('/')           # Main form
@app.route('/process')    # Submit URLs, show estimate
@app.route('/confirm')    # Start background job
@app.route('/status')     # AJAX progress updates
@app.route('/downloads')  # Serve output files
```

**Technology**:
- Flask web framework
- Threading for background jobs
- In-memory state (simple, suitable for single-user)
- Jinja2 templates

---

## Data Flow

### Complete Processing Flow (YouTube → Stems)

```
1. Input: YouTube URL
   ↓
2. YouTubeAudioDownloader.download(url)
   - Fetch metadata via yt-dlp
   - Download audio stream
   - Convert to MP3/WAV via FFmpeg
   - Sanitize filename
   ↓
3. Output: audio_file.mp3 in downloads/
   ↓
4. StemSeparator.separate(audio_file)
   - Load Demucs model (cached after first use)
   - Read audio file
   - Convert to numpy array (if needed)
   - Normalize audio
   - Resample to model.samplerate
   - Convert to PyTorch tensor
   - Move to device (CUDA/CPU)
   ↓
5. Demucs Processing
   - apply_model() with configured parameters
   - shifts=1, overlap=0.25, progress=True
   - split=True (standard) or False (high-perf)
   ↓
6. Post-Processing
   - Convert tensors back to numpy
   - Normalize each stem
   - Save as MP3/WAV via FFmpeg
   ↓
7. Output: Individual stem files
   separated/htdemucs_ft/Song Name/
   ├── vocals.mp3
   ├── drums.mp3
   ├── bass.mp3
   └── other.mp3
```

### Two-Stems Mode (Karaoke)

```
1. Input: Audio file
   ↓
2. StemSeparator.separate(audio_file, two_stems='vocals')
   ↓
3. Demucs Internal Processing
   - Separates into vocals + combined_instrumental
   - Automatically merges all non-vocal stems
   ↓
4. Output: Two files
   - vocals.mp3 (isolated vocals)
   - no_vocals.mp3 (all instruments combined)
```

### Batch Processing Flow

```
1. Input: urls.txt file
   ↓
2. Parse file
   - Read lines
   - Filter comments (#) and empty lines
   - Extract valid URLs
   ↓
3. For each URL:
   - Update progress tracker
   - Process via MusicProcessor/KaraokeCreator
   - Handle errors (log, continue)
   - Optional delay
   ↓
4. Output: All processed files
   - Success/failure summary
   - Optional error log file
```

---

## Processing Pipeline

### Audio Processing Steps

#### 1. Download Phase (yt-dlp + FFmpeg)

```python
# yt-dlp extracts best audio stream
ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': f'{output_dir}/{filename}.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',  # or 'wav'
        'preferredquality': '320'
    }]
}
```

**What happens**:
- yt-dlp fetches video metadata
- Selects best audio-only stream (or extracts from video)
- Downloads raw audio
- FFmpeg converts to target format
- Output: `downloads/Song_Name.mp3`

#### 2. Separation Phase (Demucs)

**Preprocessing**:
```python
# Read audio
wav = AudioFile(audio_path).read(
    streams=0,
    samplerate=model.samplerate,
    channels=model.audio_channels
)

# Type check (Demucs version compatibility)
if isinstance(wav, torch.Tensor):
    wav = wav.numpy()

# Normalize
ref = wav.mean(0)
wav = (wav - ref.mean()) / ref.std()
```

**Model Application**:
```python
# Convert to tensor and move to device
wav_tensor = torch.from_numpy(wav).to(device)

# Apply Demucs model
sources = apply_model(
    model,
    wav_tensor,
    shifts=1,           # 1 random shift for quality
    split=True,         # Process in chunks (standard mode)
    overlap=0.25,       # 25% overlap between chunks
    progress=True       # Show progress bar
)
```

**Post-processing**:
```python
# For each stem
for name, source in stems.items():
    # Denormalize
    source = source * ref.std() + ref.mean()

    # Save as audio file (via FFmpeg)
    if mp3:
        save_as_mp3(source, output_path, bitrate)
    else:
        save_as_wav(source, output_path)
```

### Performance Modes

#### Standard Mode (`high_performance=False`)
- `split=True`: Audio split into chunks (~10-15 second segments)
- Lower memory usage (~1-2GB VRAM)
- GPU utilization: ~5-15%
- Processing time: Baseline
- Safer for systems with limited VRAM

#### High-Performance Mode (`high_performance=True`)
- `split=False`: Entire song loaded into GPU memory
- Higher memory usage (~3-4GB VRAM for typical song)
- GPU utilization: ~80-100%
- Processing time: 2-3x faster than standard
- Requires adequate VRAM (6GB+ recommended)
- Best for: RTX 2060 and higher

---

## Web Interface

### Technology Stack

- **Backend**: Flask (Python web framework)
- **Frontend**: HTML + JavaScript (AJAX)
- **State Management**: In-memory Python dict
- **Background Jobs**: Python threading
- **File Serving**: Flask send_from_directory

### Application Flow

```
1. User visits http://localhost:5001
   ↓
2. Renders index.html form
   - URL input (textarea for multiple)
   - Mode selection (karaoke/4-stem/6-stem/download)
   - Model selection
   - High-performance toggle
   ↓
3. User submits form → POST /process
   - Parse URLs
   - Calculate estimates
   - Show confirmation page
   ↓
4. User confirms → POST /confirm
   - Start background thread with processing job
   - Return status page
   ↓
5. Frontend polls GET /status every 2 seconds
   - Returns JSON with progress
   - Updates UI dynamically
   ↓
6. Job completes
   - Status page shows completion
   - Provides download links
   ↓
7. User downloads files via /downloads/<path>
```

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Main form page |
| `/process` | POST | Submit URLs, get estimate |
| `/confirm` | POST | Start background job |
| `/status` | GET | Get current job status (JSON) |
| `/downloads/<path>` | GET | Download output files |

### Background Processing

```python
# Global state
job_status = {
    'is_running': False,
    'progress': 0,
    'total': 0,
    'current_task': 'Waiting...',
    'error': None,
    'output_files': []
}

# Background job
def run_processing_job(urls, mode, model, high_performance):
    global job_status
    job_status['is_running'] = True

    for i, url in enumerate(urls):
        job_status['current_task'] = f"Processing {i+1}/{len(urls)}"

        try:
            # Process URL
            result = processor.process(url)
            job_status['output_files'].extend(result)
        except Exception as e:
            job_status['error'] = str(e)
            break

        job_status['progress'] = i + 1

    job_status['is_running'] = False

# Start job in background thread
thread = threading.Thread(
    target=run_processing_job,
    args=(urls, mode, model, high_performance)
)
thread.daemon = True
thread.start()
```

---

## Deployment Options

### 1. Local Python Environment (Development/Personal Use)

**Setup**:
```bash
# Windows
setup.bat
venv\Scripts\activate
python music_processor.py "URL"

# Linux/Mac
./setup.sh
source venv/bin/activate
python music_processor.py "URL"
```

**Pros**:
- Full GPU support
- Fastest performance
- Direct file access
- Easy debugging

**Cons**:
- Requires local Python setup
- Manual dependency management
- Command-line only (unless running app.py)

---

### 2. Docker Container (Cross-platform)

**Setup**:
```bash
# Build image
docker-compose build

# Start container
docker-compose up -d

# Access web UI
http://localhost:5001
```

**Configuration** (`docker-compose.yml`):
```yaml
version: '3.8'
services:
  music-separator:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "5001:5001"
    volumes:
      - ./downloads:/app/downloads
      - ./separated:/app/separated
      - ./karaoke:/app/karaoke
    restart: unless-stopped
```

**Pros**:
- Isolated environment
- Consistent across platforms
- Easy deployment
- Built-in web UI

**Cons**:
- CPU-only (no GPU in standard Docker)
- Slightly slower than native
- Requires Docker installation

**GPU Support** (requires nvidia-docker):
```yaml
# Add to docker-compose.yml
services:
  music-separator:
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

---

### 3. Production Web Server

**Options**:
- **Gunicorn**: Production WSGI server
- **Nginx**: Reverse proxy + static file serving
- **Celery**: Distributed task queue (instead of threading)
- **Redis**: Job state management (instead of in-memory)

**Example Production Stack**:
```
nginx (port 80/443)
  ↓
gunicorn (multiple workers)
  ↓
Flask app
  ↓
Celery workers (processing jobs)
  ↓
Redis (job queue + state)
```

---

## Technology Stack

### Core Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| **yt-dlp** | 2025.11.12 | YouTube downloading |
| **demucs** | ≥4.0.0 | AI stem separation |
| **torch** | ≥2.0.0 | Deep learning framework |
| **torchaudio** | ≥2.0.0 | Audio I/O for PyTorch |
| **ffmpeg-python** | ≥0.2.0 | FFmpeg wrapper |
| **Flask** | ≥2.0.0 | Web framework |

### Supporting Dependencies

- **numpy**: Numerical computing
- **scipy**: Scientific computing
- **tqdm**: Progress bars
- **einops**: Tensor operations
- **julius**: Audio resampling
- **lameenc**: MP3 encoding

### External Tools

- **FFmpeg**: Audio/video processing (required)
- **NVIDIA CUDA**: GPU acceleration (optional)

### Model Files

Models are downloaded on first use and cached in:
- **Windows**: `C:\Users\USERNAME\.cache\torch\hub\checkpoints\`
- **Linux/Mac**: `~/.cache/torch/hub/checkpoints/`

Each model consists of 4 files (~80MB each, ~320MB total per model).

---

## File Organization

```
music-stem-separator/
├── Core Scripts
│   ├── youtube_downloader.py      # Download service
│   ├── stem_separator.py          # Separation service
│   ├── music_processor.py         # Orchestrator
│   ├── karaoke.py                 # Karaoke creator
│   ├── batch_processor.py         # Batch processing
│   └── smart_processor.py         # Auto-selection
│
├── Web Interface
│   ├── app.py                     # Flask application
│   └── templates/                 # HTML templates
│       └── index.html
│
├── Configuration
│   ├── requirements.txt           # Python dependencies
│   ├── setup.bat                  # Windows setup
│   ├── setup.sh                   # Linux/Mac setup
│   ├── Dockerfile                 # Docker image
│   └── docker-compose.yml         # Docker compose config
│
├── Documentation
│   ├── README.md                  # User guide
│   ├── QUICKSTART.md              # Quick start
│   ├── ARCHITECTURE.md            # This file
│   ├── PERFORMANCE_GUIDE.md       # Performance tuning
│   └── CLAUDE.md                  # AI assistant guide
│
├── Examples
│   ├── example.py                 # API usage examples
│   ├── config.example.py          # Config template
│   └── urls.example.txt           # Batch URL example
│
└── Runtime Output (created automatically)
    ├── downloads/                 # Downloaded audio
    ├── separated/                 # Separated stems
    └── karaoke/                   # Karaoke tracks
```

---

## Performance Characteristics

### Model Performance (5-minute song)

| Model | GPU (RTX 2060) | CPU (i7) | Quality | Stems |
|-------|----------------|----------|---------|-------|
| mdx_extra | ~30-45 sec | ~3-5 min | Good | 4 |
| htdemucs_ft | ~1-1.5 min | ~5-10 min | Excellent | 4 |
| htdemucs | ~1.5-2 min | ~10-15 min | Best (4-stem) | 4 |
| htdemucs_6s | ~2-3 min | ~15-25 min | Best (6-stem) | 6 |

*High-performance mode reduces GPU times by 2-3x*

### Memory Requirements

**GPU (VRAM)**:
- Standard mode: 1-2GB
- High-performance mode: 3-4GB

**System RAM**:
- Minimum: 8GB
- Recommended: 16GB
- Optimal: 32GB (for batch processing)

### Storage

**Per Song (5-minute, MP3 320kbps)**:
- Download: ~12MB
- Each stem: ~12MB
- Karaoke (2 files): ~24MB
- 4-stem: ~48MB (without original)
- 6-stem: ~72MB (without original)

---

## Security Considerations

### Input Validation

- URL validation before processing
- Filename sanitization (removes special characters)
- Path traversal prevention

### Docker Isolation

- Runs in isolated container
- Limited file system access via volumes
- No privileged mode required (unless GPU)

### Web Interface

- No authentication (designed for local/trusted use)
- For public deployment, add:
  - User authentication
  - Rate limiting
  - CSRF protection
  - Input sanitization

---

## Extensibility

### Adding New Models

```python
# In stem_separator.py
AVAILABLE_MODELS = {
    'new_model': {
        'name': 'new_model',
        'description': 'Description',
        'stems': 4
    }
}
```

### Custom Processing Modes

```python
# Create new coordinator class
class CustomProcessor:
    def __init__(self):
        self.downloader = YouTubeAudioDownloader()
        self.separator = StemSeparator()

    def custom_process(self, url):
        # Custom workflow
        pass
```

### Plugin Architecture

Potential future enhancements:
- Effect plugins (reverb, EQ, etc.)
- Format converters
- Metadata preservation
- Cloud storage integration

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| FFmpeg not found | Not in PATH | Install and add to PATH |
| CUDA out of memory | VRAM insufficient | Use `--device cpu` or standard mode |
| Model download fails | Network/cache issue | Check connection, clear cache |
| YouTube download fails | yt-dlp outdated | `pip install --upgrade yt-dlp` |
| Web UI not loading | Port conflict | Change port in app.py |

### Debug Mode

```bash
# Enable verbose logging
python music_processor.py "URL" --verbose

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Test FFmpeg
ffmpeg -version

# List available models
python stem_separator.py --list-models
```

---

## Future Enhancements

### Planned Features

- [ ] Real-time progress streaming (WebSockets)
- [ ] Persistent job queue (Celery + Redis)
- [ ] User authentication and multi-tenancy
- [ ] Cloud storage integration (S3, GCS)
- [ ] REST API for external integrations
- [ ] Mobile-responsive web UI
- [ ] Pre/post processing effects
- [ ] Metadata preservation
- [ ] Playlist support
- [ ] Resume incomplete jobs

### Performance Optimizations

- [ ] Model quantization for faster inference
- [ ] Batch GPU processing (multiple songs in parallel)
- [ ] Streaming separation (process as downloading)
- [ ] Adaptive quality based on hardware
- [ ] Multi-GPU support

---

## Conclusion

This architecture provides a solid foundation for audio stem separation with:

- **Modularity**: Each component can be used independently
- **Flexibility**: Multiple interfaces (CLI, API, Web)
- **Scalability**: From single songs to batch processing
- **Performance**: GPU acceleration with intelligent fallbacks
- **Usability**: Simple setup and intuitive interfaces

The system balances power user needs (command-line, Python API) with accessibility (web UI, Docker) while maintaining high code quality and maintainability.
