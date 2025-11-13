"""
Music Stem Separator
Separates music into individual stems (vocals, drums, bass, other) using Demucs.
Supports both GPU (NVIDIA CUDA) and CPU processing.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path

# Set console encoding for Windows
if sys.platform == 'win32':
    import io
    # Only wrap if not already wrapped
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
from demucs.pretrained import get_model, ModelLoadingError
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio


class StemSeparator:
    """Separates music into individual stems using Demucs."""

    # Available models with their characteristics
    MODELS = {
        'htdemucs': {
            'description': 'Hybrid Transformer Demucs (best quality, slower)',
            'stems': ['drums', 'bass', 'other', 'vocals'],
            'best_for': 'highest quality separation'
        },
        'htdemucs_ft': {
            'description': 'Fine-tuned Hybrid Transformer Demucs (best overall)',
            'stems': ['drums', 'bass', 'other', 'vocals'],
            'best_for': 'best balance of quality and speed'
        },
        'htdemucs_6s': {
            'description': '6-stem model (most detailed)',
            'stems': ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano'],
            'best_for': 'detailed instrument separation'
        },
        'mdx_extra': {
            'description': 'MDX-Net model (fast, good quality)',
            'stems': ['drums', 'bass', 'other', 'vocals'],
            'best_for': 'faster processing with good quality'
        }
    }

    def __init__(self, model_name='htdemucs_ft', device=None, output_dir='separated', high_performance=False):
        """
        Initialize the stem separator.

        Args:
            model_name (str): Model to use for separation
            device (str): Device to use ('cuda', 'cpu', or None for auto-detect)
            output_dir (str): Directory to save separated stems
            high_performance (bool): Enable high-performance GPU mode (requires more VRAM)
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.high_performance = high_performance

        # Detect and set device
        if device is None:
            self.device = self._detect_device()
        else:
            self.device = device

        print(f"Using device: {self.device}")
        if high_performance and self.device == 'cuda':
            print(f"High-performance mode: ENABLED (maximum GPU utilization)")

        # Check CUDA availability for GPU
        if self.device == 'cuda':
            if not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
                self.device = 'cpu'
            else:
                print(f"CUDA available: {torch.cuda.get_device_name(0)}")
                print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Load the model
        print(f"Loading model: {model_name}")
        try:
            self.model = get_model(model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"Model loaded successfully")
            print(f"Available stems: {', '.join(self.model.sources)}")
        except ModelLoadingError as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            raise

    @staticmethod
    def _detect_device():
        """Detect best available device (CUDA GPU or CPU)."""
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

    def separate(self, audio_file, two_stems=None, mp3=True, mp3_bitrate=320, float32=False, int24=False):
        """
        Separate audio file into stems.

        Args:
            audio_file (str): Path to audio file
            two_stems (str, optional): Extract only two stems (e.g., 'vocals' splits into vocals/no_vocals)
            mp3 (bool): Save as MP3 (True) or WAV (False)
            mp3_bitrate (int): MP3 bitrate in kbps
            float32 (bool): Save as 32-bit float WAV
            int24 (bool): Save as 24-bit int WAV

        Returns:
            dict: Dictionary mapping stem names to file paths
        """
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        print(f"\nSeparating: {audio_path.name}")
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")

        # Create output directory for this file
        song_name = audio_path.stem
        song_output_dir = self.output_dir / self.model_name / song_name
        song_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load audio
            print("Loading audio...")
            wav = AudioFile(str(audio_path)).read(
                streams=0,
                samplerate=self.model.samplerate,
                channels=self.model.audio_channels
            )

            # Convert to numpy if it's a tensor
            if isinstance(wav, torch.Tensor):
                wav = wav.numpy()

            # Normalize audio
            ref = wav.mean(0)
            wav = (wav - ref.mean()) / ref.std()

            # Convert to tensor and move to device
            wav_tensor = torch.from_numpy(wav).to(self.device)

            # Add batch dimension if needed
            if wav_tensor.dim() == 2:
                wav_tensor = wav_tensor.unsqueeze(0)

            # Apply model
            # Configure processing parameters based on performance mode
            if self.high_performance and self.device == 'cuda':
                # High-performance: Load entire song into GPU, use more shifts
                shifts = 1  # Keep at 1 for speed; increase to 5-10 for better quality
                split = False  # Don't split - load entire song for maximum GPU usage
                print("Separating stems with HIGH-PERFORMANCE GPU mode...")
                print("  (Using full GPU memory for maximum speed)")
            else:
                # Standard: Split into chunks to save memory
                shifts = 1
                split = True
                print("Separating stems (this may take several minutes)...")

            with torch.no_grad():
                sources = apply_model(
                    self.model,
                    wav_tensor,
                    device=self.device,
                    shifts=shifts,
                    split=split,
                    overlap=0.25,
                    progress=True
                )

            # Denormalize and move results back to CPU
            sources = sources * ref.std() + ref.mean()
            sources = sources.cpu()

            # Prepare stems dictionary
            stems_files = {}

            # Handle two-stem mode
            if two_stems is not None:
                if two_stems not in self.model.sources:
                    raise ValueError(f"Invalid stem: {two_stems}. Available: {self.model.sources}")

                stem_index = self.model.sources.index(two_stems)
                stem_source = sources[0, stem_index]
                other_source = sources[0].sum(0) - stem_source

                # Save the selected stem
                stem_file = self._save_stem(
                    song_output_dir, song_name, two_stems, stem_source,
                    mp3, mp3_bitrate, float32, int24
                )
                stems_files[two_stems] = stem_file.as_posix()

                # Save the rest as "no_{stem}"
                other_name = f"no_{two_stems}"
                other_file = self._save_stem(
                    song_output_dir, song_name, other_name, other_source,
                    mp3, mp3_bitrate, float32, int24
                )
                stems_files[other_name] = other_file.as_posix()

            else:
                # Save all stems
                for stem_index, stem_name in enumerate(self.model.sources):
                    stem_source = sources[0, stem_index]
                    stem_file = self._save_stem(
                        song_output_dir, song_name, stem_name, stem_source,
                        mp3, mp3_bitrate, float32, int24
                    )
                    stems_files[stem_name] = stem_file.as_posix()

            print(f"\n✓ Separation completed!")
            print(f"  Output directory: {song_output_dir}")
            print(f"  Stems created: {len(stems_files)}")

            return stems_files

        except Exception as e:
            print(f"\n✗ Error during separation: {str(e)}", file=sys.stderr)
            raise

    def _save_stem(self, output_dir, song_name, stem_name, stem_audio, mp3, mp3_bitrate, float32, int24):
        """Save individual stem to file."""
        if mp3:
            ext = 'mp3'
        else:
            ext = 'wav'

        stem_file = output_dir / f"{stem_name}.{ext}"

        print(f"  Saving {stem_name}...", end=' ')

        save_audio(
            stem_audio,
            str(stem_file),
            samplerate=self.model.samplerate,
            bitrate=mp3_bitrate if mp3 else None,
            clip='rescale',
            as_float=float32,
            bits_per_sample=24 if int24 else 16
        )

        print(f"✓ ({stem_file.name})")
        return stem_file

    @classmethod
    def list_models(cls):
        """Print available models and their characteristics."""
        print("\nAvailable Models:")
        print("=" * 80)
        for model_name, info in cls.MODELS.items():
            print(f"\n{model_name}")
            print(f"  Description: {info['description']}")
            print(f"  Stems: {', '.join(info['stems'])}")
            print(f"  Best for: {info['best_for']}")
        print("\n" + "=" * 80)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Separate music into individual stems using Demucs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Separate into all stems (vocals, drums, bass, other)
  python stem_separator.py song.mp3

  # Use 6-stem model for detailed separation
  python stem_separator.py song.mp3 --model htdemucs_6s

  # Extract only vocals (creates vocals and no_vocals)
  python stem_separator.py song.mp3 --two-stems vocals

  # Use CPU instead of GPU
  python stem_separator.py song.mp3 --device cpu

  # Save as WAV instead of MP3
  python stem_separator.py song.mp3 --wav

  # List available models
  python stem_separator.py --list-models
        """
    )

    parser.add_argument(
        'audio_file',
        nargs='?',
        help='Audio file to separate'
    )
    parser.add_argument(
        '--model', '-m',
        default='htdemucs_ft',
        help='Model to use (default: htdemucs_ft)'
    )
    parser.add_argument(
        '--device', '-d',
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='separated',
        help='Output directory (default: separated)'
    )
    parser.add_argument(
        '--two-stems',
        choices=['vocals', 'drums', 'bass', 'other', 'guitar', 'piano'],
        help='Extract only two stems (selected stem and everything else)'
    )
    parser.add_argument(
        '--wav',
        action='store_true',
        help='Save as WAV instead of MP3'
    )
    parser.add_argument(
        '--mp3-bitrate',
        type=int,
        default=320,
        help='MP3 bitrate in kbps (default: 320)'
    )
    parser.add_argument(
        '--float32',
        action='store_true',
        help='Save as 32-bit float WAV'
    )
    parser.add_argument(
        '--int24',
        action='store_true',
        help='Save as 24-bit int WAV'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit'
    )
    parser.add_argument(
        '--high-performance',
        action='store_true',
        help='Enable high-performance GPU mode (uses more VRAM but faster)'
    )

    args = parser.parse_args()

    # Handle list models command
    if args.list_models:
        StemSeparator.list_models()
        return 0

    # Validate audio file is provided
    if not args.audio_file:
        parser.error("audio_file is required unless using --list-models")

    try:
        separator = StemSeparator(
            model_name=args.model,
            device=args.device,
            output_dir=args.output_dir,
            high_performance=args.high_performance
        )

        stems = separator.separate(
            audio_file=args.audio_file,
            two_stems=args.two_stems,
            mp3=not args.wav,
            mp3_bitrate=args.mp3_bitrate,
            float32=args.float32,
            int24=args.int24
        )

        print("\nSeparated stems:")
        for stem_name, stem_path in stems.items():
            print(f"  {stem_name}: {stem_path}")

        return 0

    except Exception as e:
        print(f"\nFailed to separate stems: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
