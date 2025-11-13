"""
Configuration File (Optional)
Copy this file to config.py and modify the settings as needed.

Usage:
  from config import CONFIG
  processor = MusicProcessor(**CONFIG['processor'])
"""

CONFIG = {
    # YouTube Downloader Settings
    'downloader': {
        'output_dir': 'downloads',
        'format': 'mp3',  # 'mp3' or 'wav'
        'quality': '320',  # MP3 bitrate in kbps
    },

    # Stem Separator Settings
    'separator': {
        'model_name': 'htdemucs_ft',  # 'htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'mdx_extra'
        'device': None,  # None for auto-detect, 'cuda' for GPU, 'cpu' for CPU
        'output_dir': 'separated',
    },

    # Music Processor Settings
    'processor': {
        'download_dir': 'downloads',
        'output_dir': 'separated',
        'audio_format': 'mp3',
        'model': 'htdemucs_ft',
        'device': None,  # Auto-detect
    },

    # Output Settings
    'output': {
        'stem_format': 'mp3',  # 'mp3' or 'wav'
        'mp3_bitrate': 320,  # MP3 bitrate in kbps
        'keep_original': True,  # Keep downloaded file after separation
    },

    # Processing Settings
    'processing': {
        'two_stems': None,  # None for all stems, or 'vocals', 'drums', 'bass', etc.
        'float32': False,  # Use 32-bit float WAV
        'int24': False,  # Use 24-bit int WAV
    },
}

# Model descriptions for reference
MODELS = {
    'htdemucs': 'Hybrid Transformer Demucs - Best quality, slowest',
    'htdemucs_ft': 'Fine-tuned Hybrid Transformer - Best overall (recommended)',
    'htdemucs_6s': '6-stem model - Most detailed separation',
    'mdx_extra': 'MDX-Net - Fastest, good quality',
}

# Stem options for reference
AVAILABLE_STEMS = {
    '4-stem': ['vocals', 'drums', 'bass', 'other'],
    '6-stem': ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other'],
}
