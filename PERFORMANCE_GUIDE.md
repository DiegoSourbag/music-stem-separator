# Performance Guide

Guide to optimizing performance and selecting the right quality mode.

## Model Download & Caching

### How Model Caching Works

Models are downloaded **once** and cached permanently:

```
C:\Users\USERNAME\.cache\torch\hub\checkpoints\
├── htdemucs_ft.th           (~80 MB × 4 files = ~320 MB)
├── htdemucs_6s.th           (~80 MB × 4 files = ~320 MB)
├── htdemucs.th              (~80 MB × 4 files = ~320 MB)
└── mdx_extra.th             (~80 MB × 4 files = ~320 MB)
```

**Key Points:**
- ✅ Downloaded once, cached forever
- ✅ Shared across all projects
- ✅ Instant loading after first download
- ✅ Total space for all models: ~1.3 GB
- ✅ Only downloads what you actually use

### First Run vs Subsequent Runs

**First time using a model:**
1. Downloads model files (~320 MB, takes 1-5 minutes depending on connection)
2. Loads into memory
3. Processes audio

**Every time after:**
1. Loads from cache (instant)
2. Processes audio

## Quality Modes Explained

### Fast Mode (`mdx_extra`)
- **Speed**: 1x (baseline - fastest)
- **Quality**: Good
- **Stems**: 4 (vocals, drums, bass, other)
- **GPU time**: ~30 sec per 5-min song
- **CPU time**: ~3-5 min per 5-min song

**Best for:**
- Batch processing many songs
- CPU-only systems
- Quick results needed
- Background/karaoke tracks

**Trade-offs:**
- Slightly less precise separation
- Vocals may have slight instrument bleed
- Good enough for most casual uses

### Balanced Mode (`htdemucs_ft`) - **RECOMMENDED**
- **Speed**: 2x
- **Quality**: Excellent
- **Stems**: 4 (vocals, drums, bass, other)
- **GPU time**: ~1 min per 5-min song
- **CPU time**: ~5-10 min per 5-min song

**Best for:**
- General use
- Best quality/speed balance
- Most projects
- Karaoke creation

**Trade-offs:**
- Takes twice as long as fast mode
- Still 4-stem (no separate guitar/piano)

### Quality Mode (`htdemucs`)
- **Speed**: 3x
- **Quality**: Best (4-stem)
- **Stems**: 4 (vocals, drums, bass, other)
- **GPU time**: ~1.5 min per 5-min song
- **CPU time**: ~10-15 min per 5-min song

**Best for:**
- Professional audio work
- Maximum 4-stem quality
- Clean vocal isolation
- Studio projects

**Trade-offs:**
- Takes 3x longer than fast mode
- Still 4-stem (no guitar/piano separation)
- Minimal improvement over balanced for casual use

### Detailed Mode (`htdemucs_6s`)
- **Speed**: 4x (slowest)
- **Quality**: Best + Most stems
- **Stems**: 6 (vocals, drums, bass, **guitar**, **piano**, other)
- **GPU time**: ~2-3 min per 5-min song
- **CPU time**: ~15-25 min per 5-min song

**Best for:**
- Need separate guitar tracks
- Need separate piano tracks
- Detailed remixing
- Professional stem separation

**Trade-offs:**
- Takes 4x longer than fast mode
- Larger model download (~30% bigger)
- Only needed if you specifically want guitar/piano separated

## Smart Processor Auto-Selection Logic

When using `--quality auto`, the smart processor selects:

```
IF --need-guitar OR --need-piano:
    → DETAILED mode (htdemucs_6s)
    Reason: Guitar/piano separation required

ELSE IF --prefer-speed:
    → FAST mode (mdx_extra)
    Reason: Speed prioritized

ELSE IF device == 'cpu':
    → FAST mode (mdx_extra)
    Reason: CPU mode - save time

ELSE:
    → BALANCED mode (htdemucs_ft)
    Reason: Best for general use
```

## Hardware Recommendations

### GPU Processing (NVIDIA CUDA)

**Minimum:**
- GPU: GTX 1050 Ti or better
- VRAM: 4 GB
- Can process: Fast and Balanced modes comfortably

**Recommended:**
- GPU: RTX 2060 or better (like yours!)
- VRAM: 6+ GB
- Can process: All modes efficiently

**Optimal:**
- GPU: RTX 3060 or better
- VRAM: 8+ GB
- Can process: All modes, including batch processing

### CPU Processing

**Minimum:**
- CPU: i5 or Ryzen 5 (quad-core)
- RAM: 8 GB
- Recommended mode: Fast

**Recommended:**
- CPU: i7 or Ryzen 7 (8+ cores)
- RAM: 16 GB
- Can use: Fast and Balanced modes

**Optimal:**
- CPU: i9 or Ryzen 9 (12+ cores)
- RAM: 32 GB
- Can use: All modes (but still slow compared to GPU)

## Batch Processing Strategies

### Small Batch (1-10 songs)
```bash
# Use balanced mode for quality
python.exe batch_processor.py urls.txt --model htdemucs_ft
```

### Medium Batch (10-50 songs)
```bash
# Use fast mode to save time
python.exe batch_processor.py urls.txt --model mdx_extra --delay 3
```

### Large Batch (50+ songs)
```bash
# Definitely use fast mode, add delays to avoid rate limiting
python.exe batch_processor.py urls.txt --model mdx_extra --delay 5
```

### CPU Batch Processing
```bash
# Always use fast mode on CPU
python.exe batch_processor.py urls.txt --model mdx_extra --device cpu
```

## Estimating Processing Time

### Formula
```
Total Time = Download Time + (Number of Songs × Processing Time per Song)

Processing Time per Song = Song Duration × Model Speed Factor
```

### Example: 20 songs, average 4 minutes each

**GPU (RTX 2060):**
- Fast: 20 × 4 min × 0.125 = ~10 minutes processing
- Balanced: 20 × 4 min × 0.25 = ~20 minutes processing
- Detailed: 20 × 4 min × 0.5 = ~40 minutes processing

**CPU (i7):**
- Fast: 20 × 4 min × 0.75 = ~60 minutes processing
- Balanced: 20 × 4 min × 1.5 = ~120 minutes processing
- Detailed: 20 × 4 min × 3.0 = ~240 minutes processing

Plus download time (~1-2 min per song)

## Tips for Optimization

### 1. Use Appropriate Quality Mode
```bash
# Don't use detailed mode if you don't need guitar/piano
python.exe smart_processor.py "URL" --quality balanced  # Good enough for most

# Only use detailed when needed
python.exe smart_processor.py "URL" --need-guitar --mode full
```

### 2. Batch Processing on GPU
```bash
# Process overnight with detailed mode
python.exe batch_processor.py urls.txt --model htdemucs_6s
```

### 3. CPU + Batch = Use Fast
```bash
# On CPU, fast mode is the only practical choice for batches
python.exe batch_processor.py urls.txt --model mdx_extra --device cpu
```

### 4. Test First
```bash
# Try fast mode first to see if quality is acceptable
python.exe smart_processor.py "URL" --quality fast

# If not satisfied, re-run with better model (download is cached!)
python.exe smart_processor.py "URL" --quality balanced
```

### 5. Parallel Processing (Advanced)
If you have multiple GPUs or want to use CPU cores efficiently:
- Split URL list into multiple files
- Run multiple batch_processor instances in separate terminals
- Each uses a different GPU or CPU cores

### 6. Don't Keep Original Downloads
```bash
# Save disk space when batch processing
python.exe batch_processor.py urls.txt --no-keep-original
```

## Quality Comparison

### When to Use Each Mode

| Your Goal | Recommended Mode | Reason |
|-----------|------------------|---------|
| Quick karaoke for party | Fast | Good enough, saves time |
| General karaoke/instrumental | Balanced | Best quality/time balance |
| Professional vocal removal | Quality | Cleanest separation |
| Isolate guitar solo | Detailed | Only mode that separates guitar |
| Remix with separate piano | Detailed | Only mode that separates piano |
| 100 songs for playlist | Fast | Only practical choice |
| Learning/practice | Fast or Balanced | Good enough, faster iteration |
| Studio/commercial use | Quality or Detailed | Professional quality needed |

## Storage Requirements

### Per Song (5-minute song, 320 kbps MP3)

**Karaoke mode:**
- Original download: ~12 MB
- Instrumental: ~12 MB
- Vocals (if kept): ~12 MB
- **Total**: ~24-36 MB per song

**Full separation (4-stem):**
- Original download: ~12 MB
- Each stem (×4): ~12 MB each
- **Total**: ~60 MB per song (or ~48 MB if original deleted)

**Full separation (6-stem):**
- Original download: ~12 MB
- Each stem (×6): ~12 MB each
- **Total**: ~84 MB per song (or ~72 MB if original deleted)

### Batch Processing Storage

**100 songs, karaoke mode:**
- ~2.4-3.6 GB

**100 songs, 4-stem:**
- ~4.8-6.0 GB

**100 songs, 6-stem:**
- ~7.2-8.4 GB

## Conclusion

**For most users:**
- Use **Smart Processor** with auto mode
- It will pick the right model based on your needs
- Models download once and are cached
- Start with balanced mode, adjust if needed

**Quick Decision Guide:**
```bash
# "I just want karaoke tracks quickly"
python.exe smart_processor.py "URL" --quality fast

# "I want good quality without waiting too long"
python.exe smart_processor.py "URL"  # Uses auto → balanced

# "I need to separate guitar"
python.exe smart_processor.py "URL" --need-guitar --mode full

# "I have 50 songs to process on CPU"
python.exe batch_processor.py urls.txt --model mdx_extra --device cpu
```
