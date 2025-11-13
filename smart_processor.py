"""
Smart Music Processor
Automatically selects the best model and settings based on requirements and hardware.
"""

import sys
import argparse
import torch
import yt_dlp
from pathlib import Path
from music_processor import MusicProcessor
from karaoke import KaraokeCreator


class SmartProcessor:
    """
    Intelligently selects models and settings based on requirements.

    Quality Modes:
    - fast: Quick processing, good quality (mdx_extra)
    - balanced: Best quality/speed trade-off (htdemucs_ft) - DEFAULT
    - quality: Highest quality 4-stem (htdemucs)
    - detailed: Full 6-stem separation (htdemucs_6s)
    """

    QUALITY_MODES = {
        'fast': {
            'model': 'mdx_extra',
            'description': 'Fast processing, good quality',
            'stems': 4,
            'relative_speed': '1x (fastest)',
            'best_for': 'Quick results, batch processing',
            'gpu_factor': 0.125,  # GPU: ~7.5 sec per minute of audio
            'cpu_factor': 0.75    # CPU: ~45 sec per minute of audio
        },
        'balanced': {
            'model': 'htdemucs_ft',
            'description': 'Best balance of quality and speed',
            'stems': 4,
            'relative_speed': '2x',
            'best_for': 'Most use cases (recommended)',
            'gpu_factor': 0.25,   # GPU: ~15 sec per minute of audio
            'cpu_factor': 1.5     # CPU: ~90 sec per minute of audio
        },
        'quality': {
            'model': 'htdemucs',
            'description': 'Highest quality 4-stem separation',
            'stems': 4,
            'relative_speed': '3x',
            'best_for': 'Maximum quality without extra stems',
            'gpu_factor': 0.375,  # GPU: ~22.5 sec per minute of audio
            'cpu_factor': 2.25    # CPU: ~135 sec per minute of audio
        },
        'detailed': {
            'model': 'htdemucs_6s',
            'description': 'Full 6-stem separation (includes guitar, piano)',
            'stems': 6,
            'relative_speed': '4x (slowest)',
            'best_for': 'Detailed instrument separation',
            'gpu_factor': 0.5,    # GPU: ~30 sec per minute of audio
            'cpu_factor': 3.0     # CPU: ~180 sec per minute of audio
        }
    }

    def __init__(self, quality='auto', device=None, verbose=True):
        """
        Initialize smart processor.

        Args:
            quality (str): Quality mode ('auto', 'fast', 'balanced', 'quality', 'detailed')
            device (str): Device to use ('cuda', 'cpu', or None for auto-detect)
            verbose (bool): Print detailed information
        """
        self.quality = quality
        self.device = device if device else self._detect_device()
        self.verbose = verbose

        if self.verbose:
            print("=" * 80)
            print("SMART MUSIC PROCESSOR")
            print("=" * 80)
            print(f"Device: {self.device}")
            if self.device == 'cuda':
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print()

    @staticmethod
    def _detect_device():
        """Detect best available device."""
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def _format_time(seconds):
        """Format seconds into human-readable time."""
        if seconds < 60:
            return f"{int(seconds)} seconds"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes} min {secs} sec"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours} hr {minutes} min"

    @staticmethod
    def get_youtube_duration(url):
        """
        Get duration of YouTube video in seconds.

        Args:
            url (str): YouTube URL

        Returns:
            int: Duration in seconds (or None if failed)
        """
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info.get('duration', None)
        except Exception:
            return None

    @staticmethod
    def get_audio_duration(file_path):
        """
        Get duration of audio file in seconds.

        Args:
            file_path (str): Path to audio file

        Returns:
            int: Duration in seconds (or None if failed)
        """
        try:
            from demucs.audio import AudioFile
            audio = AudioFile(file_path).read(streams=0)
            # Assume standard sample rate of 44100 Hz
            duration = audio.shape[-1] / 44100
            return int(duration)
        except Exception:
            return None

    def estimate_processing_time(self, duration_seconds, model_name):
        """
        Estimate processing time for given audio duration and model.

        Args:
            duration_seconds (int): Audio duration in seconds
            model_name (str): Model name

        Returns:
            dict: Estimation details
        """
        if duration_seconds is None:
            return None

        # Find quality mode for this model
        quality_mode = None
        for mode, info in self.QUALITY_MODES.items():
            if info['model'] == model_name:
                quality_mode = mode
                break

        if not quality_mode:
            return None

        mode_info = self.QUALITY_MODES[quality_mode]
        duration_minutes = duration_seconds / 60.0

        # Select factor based on device
        if self.device == 'cuda':
            factor = mode_info['gpu_factor']
            device_name = "GPU"
        else:
            factor = mode_info['cpu_factor']
            device_name = "CPU"

        # Calculate times
        processing_time = duration_minutes * factor * 60  # Convert back to seconds
        download_time = 60 if duration_seconds > 0 else 0  # Estimate ~1 min for download
        total_time = processing_time + download_time

        return {
            'audio_duration': duration_seconds,
            'audio_duration_formatted': self._format_time(duration_seconds),
            'processing_time': processing_time,
            'processing_time_formatted': self._format_time(processing_time),
            'download_time': download_time,
            'download_time_formatted': self._format_time(download_time),
            'total_time': total_time,
            'total_time_formatted': self._format_time(total_time),
            'device': device_name,
            'model': model_name,
            'quality_mode': quality_mode
        }

    def show_confirmation(self, url=None, file_path=None, mode='karaoke',
                         model_name=None, estimate=None, auto_confirm=False):
        """
        Show processing summary and ask for confirmation.

        Args:
            url (str): YouTube URL (if applicable)
            file_path (str): Local file path (if applicable)
            mode (str): Processing mode
            model_name (str): Model to be used
            estimate (dict): Time estimation details
            auto_confirm (bool): Skip confirmation and proceed

        Returns:
            bool: True if confirmed, False if cancelled
        """
        print("\n" + "=" * 80)
        print("PROCESSING SUMMARY")
        print("=" * 80)

        # Input source
        if url:
            print(f"Source: YouTube")
            print(f"URL: {url}")
        else:
            print(f"Source: Local File")
            print(f"File: {file_path}")

        # Processing details
        print(f"\nMode: {mode.upper()}")
        print(f"Model: {model_name}")

        # Find quality mode
        quality_mode = None
        for mode_key, info in self.QUALITY_MODES.items():
            if info['model'] == model_name:
                quality_mode = mode_key
                stems = info['stems']
                break

        if quality_mode:
            print(f"Quality: {quality_mode.upper()}")
            print(f"Stems: {stems}")

        print(f"Device: {self.device.upper()}")

        # Time estimate
        if estimate:
            print(f"\n--- TIME ESTIMATE ---")
            print(f"Audio Duration: {estimate['audio_duration_formatted']}")
            if url:
                print(f"Download Time: ~{estimate['download_time_formatted']}")
            print(f"Processing Time: ~{estimate['processing_time_formatted']} ({estimate['device']})")
            print(f"Total Time: ~{estimate['total_time_formatted']}")
            print(f"\nNote: First-time use downloads the model (~320 MB, 1-5 min)")
        else:
            print(f"\nTime Estimate: Unable to calculate")

        print("\n" + "=" * 80)

        # Auto-confirm?
        if auto_confirm:
            print("Auto-confirmed (--yes flag)")
            return True

        # Ask for confirmation
        try:
            response = input("\nProceed with processing? [Y/n]: ").strip().lower()
            if response == '' or response == 'y' or response == 'yes':
                return True
            else:
                print("\nProcessing cancelled by user.")
                return False
        except (KeyboardInterrupt, EOFError):
            print("\n\nProcessing cancelled by user.")
            return False

    def select_model(self, need_guitar=False, need_piano=False, prefer_speed=False):
        """
        Intelligently select the best model based on requirements.

        Args:
            need_guitar (bool): Need separate guitar track
            need_piano (bool): Need separate piano track
            prefer_speed (bool): Prioritize speed over quality

        Returns:
            str: Selected model name
        """
        # If quality mode is specified (not 'auto'), use it
        if self.quality != 'auto':
            model = self.QUALITY_MODES[self.quality]['model']
            if self.verbose:
                print(f"Quality mode: {self.quality}")
                print(f"Selected model: {model}")
                print(f"Description: {self.QUALITY_MODES[self.quality]['description']}")
                print(f"Processing speed: {self.QUALITY_MODES[self.quality]['relative_speed']}")
                print()
            return model

        # Auto mode: intelligent selection
        if self.verbose:
            print("Auto mode: Analyzing requirements...")

        # Need detailed separation?
        if need_guitar or need_piano:
            selected = 'detailed'
            reason = "Guitar/piano separation required"
        # Prefer speed?
        elif prefer_speed:
            selected = 'fast'
            reason = "Speed prioritized"
        # CPU mode - use faster model
        elif self.device == 'cpu':
            selected = 'fast'
            reason = "CPU mode - using faster model"
        # Default: balanced
        else:
            selected = 'balanced'
            reason = "Best balance for general use"

        model = self.QUALITY_MODES[selected]['model']

        if self.verbose:
            print(f"Auto-selected: {selected}")
            print(f"Reason: {reason}")
            print(f"Model: {model}")
            print(f"Stems: {self.QUALITY_MODES[selected]['stems']}")
            print(f"Speed: {self.QUALITY_MODES[selected]['relative_speed']}")
            print()

        return model

    def process_youtube(self, url, mode='karaoke', need_guitar=False,
                       need_piano=False, prefer_speed=False, auto_confirm=False, **kwargs):
        """
        Process YouTube URL with smart model selection.

        Args:
            url (str): YouTube URL
            mode (str): 'karaoke' or 'full'
            need_guitar (bool): Need separate guitar track
            need_piano (bool): Need separate piano track
            prefer_speed (bool): Prioritize speed over quality
            auto_confirm (bool): Skip confirmation prompt
            **kwargs: Additional arguments for processor

        Returns:
            Result from processor
        """
        # Select appropriate model
        model = self.select_model(need_guitar, need_piano, prefer_speed)

        # Get duration and estimate processing time
        if self.verbose:
            print("Fetching video information...")
        duration = self.get_youtube_duration(url)
        estimate = self.estimate_processing_time(duration, model)

        # Show confirmation
        if not self.show_confirmation(
            url=url,
            mode=mode,
            model_name=model,
            estimate=estimate,
            auto_confirm=auto_confirm
        ):
            return None  # User cancelled

        # Process based on mode
        if mode == 'karaoke':
            processor = KaraokeCreator(
                model=model,
                device=self.device,
                output_dir=kwargs.get('output_dir', 'karaoke')
            )
            return processor.create_from_youtube(
                url=url,
                keep_vocals=kwargs.get('keep_vocals', True),
                keep_original=kwargs.get('keep_original', False),
                output_format=kwargs.get('output_format', 'mp3'),
                bitrate=kwargs.get('bitrate', 320)
            )
        else:  # full
            processor = MusicProcessor(
                model=model,
                device=self.device,
                output_dir=kwargs.get('output_dir', 'separated')
            )
            return processor.process_from_youtube(
                url=url,
                two_stems=kwargs.get('two_stems'),
                output_format=kwargs.get('output_format', 'mp3'),
                mp3_bitrate=kwargs.get('bitrate', 320),
                keep_original=kwargs.get('keep_original', False)
            )

    def process_local(self, audio_file, mode='karaoke', need_guitar=False,
                     need_piano=False, prefer_speed=False, auto_confirm=False, **kwargs):
        """
        Process local file with smart model selection.

        Args:
            audio_file (str): Path to audio file
            mode (str): 'karaoke' or 'full'
            need_guitar (bool): Need separate guitar track
            need_piano (bool): Need separate piano track
            prefer_speed (bool): Prioritize speed over quality
            auto_confirm (bool): Skip confirmation prompt
            **kwargs: Additional arguments for processor

        Returns:
            Result from processor
        """
        # Select appropriate model
        model = self.select_model(need_guitar, need_piano, prefer_speed)

        # Get duration and estimate processing time
        if self.verbose:
            print("Analyzing audio file...")
        duration = self.get_audio_duration(audio_file)
        estimate = self.estimate_processing_time(duration, model)

        # Show confirmation
        if not self.show_confirmation(
            file_path=audio_file,
            mode=mode,
            model_name=model,
            estimate=estimate,
            auto_confirm=auto_confirm
        ):
            return None  # User cancelled

        # Process based on mode
        if mode == 'karaoke':
            processor = KaraokeCreator(
                model=model,
                device=self.device,
                output_dir=kwargs.get('output_dir', 'karaoke')
            )
            return processor.create_from_file(
                audio_file=audio_file,
                keep_vocals=kwargs.get('keep_vocals', True),
                output_format=kwargs.get('output_format', 'mp3'),
                bitrate=kwargs.get('bitrate', 320)
            )
        else:  # full
            processor = MusicProcessor(
                model=model,
                device=self.device,
                output_dir=kwargs.get('output_dir', 'separated')
            )
            return processor.process_local_file(
                audio_file=audio_file,
                two_stems=kwargs.get('two_stems'),
                output_format=kwargs.get('output_format', 'mp3'),
                mp3_bitrate=kwargs.get('bitrate', 320)
            )

    @classmethod
    def print_modes(cls):
        """Print available quality modes."""
        print("\n" + "=" * 80)
        print("AVAILABLE QUALITY MODES")
        print("=" * 80)
        for mode, info in cls.QUALITY_MODES.items():
            print(f"\n{mode.upper()}")
            print(f"  Model: {info['model']}")
            print(f"  Description: {info['description']}")
            print(f"  Stems: {info['stems']}")
            print(f"  Speed: {info['relative_speed']}")
            print(f"  Best for: {info['best_for']}")
        print("\n" + "=" * 80)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Smart music processor with automatic model selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Quality Modes:
  auto      - Automatically select best model based on requirements
  fast      - Quick processing, good quality (mdx_extra)
  balanced  - Best balance of quality/speed (htdemucs_ft) - DEFAULT
  quality   - Highest quality 4-stem (htdemucs)
  detailed  - Full 6-stem with guitar/piano (htdemucs_6s)

Examples:
  # Auto mode - picks best model automatically
  python smart_processor.py "URL"

  # Fast mode for quick results
  python smart_processor.py "URL" --quality fast

  # Auto mode with guitar separation (will auto-select 6-stem)
  python smart_processor.py "URL" --need-guitar

  # Full stem separation with detailed mode
  python smart_processor.py "URL" --mode full --quality detailed

  # Process local file, prefer speed
  python smart_processor.py --local song.mp3 --prefer-speed

  # List all quality modes
  python smart_processor.py --list-modes
        """
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        'url',
        nargs='?',
        help='YouTube URL to process'
    )
    input_group.add_argument(
        '--local', '-l',
        metavar='FILE',
        help='Process local audio file'
    )

    # Quality/Model selection
    parser.add_argument(
        '--quality', '-q',
        choices=['auto', 'fast', 'balanced', 'quality', 'detailed'],
        default='auto',
        help='Quality mode (default: auto)'
    )
    parser.add_argument(
        '--need-guitar',
        action='store_true',
        help='Need separate guitar track (auto-selects 6-stem model)'
    )
    parser.add_argument(
        '--need-piano',
        action='store_true',
        help='Need separate piano track (auto-selects 6-stem model)'
    )
    parser.add_argument(
        '--prefer-speed',
        action='store_true',
        help='Prioritize speed over quality (auto mode only)'
    )

    # Processing options
    parser.add_argument(
        '--mode', '-m',
        choices=['karaoke', 'full'],
        default='karaoke',
        help='Processing mode (default: karaoke)'
    )
    parser.add_argument(
        '--device', '-d',
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory'
    )
    parser.add_argument(
        '--format', '-f',
        choices=['mp3', 'wav'],
        default='mp3',
        help='Output format (default: mp3)'
    )
    parser.add_argument(
        '--bitrate', '-b',
        type=int,
        default=320,
        help='MP3 bitrate in kbps (default: 320)'
    )
    parser.add_argument(
        '--no-vocals',
        action='store_true',
        help="Don't save vocals (karaoke mode only)"
    )
    parser.add_argument(
        '--keep-original',
        action='store_true',
        help='Keep original download'
    )
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt and proceed automatically'
    )
    parser.add_argument(
        '--list-modes',
        action='store_true',
        help='List all quality modes and exit'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )

    args = parser.parse_args()

    # List modes
    if args.list_modes:
        SmartProcessor.print_modes()
        return 0

    try:
        # Initialize smart processor
        processor = SmartProcessor(
            quality=args.quality,
            device=args.device,
            verbose=not args.quiet
        )

        # Prepare kwargs
        kwargs = {
            'output_dir': args.output_dir,
            'output_format': args.format,
            'bitrate': args.bitrate,
            'keep_vocals': not args.no_vocals,
            'keep_original': args.keep_original
        }

        # Process
        if args.local:
            result = processor.process_local(
                audio_file=args.local,
                mode=args.mode,
                need_guitar=args.need_guitar,
                need_piano=args.need_piano,
                prefer_speed=args.prefer_speed,
                auto_confirm=args.yes,
                **kwargs
            )
        else:
            result = processor.process_youtube(
                url=args.url,
                mode=args.mode,
                need_guitar=args.need_guitar,
                need_piano=args.need_piano,
                prefer_speed=args.prefer_speed,
                auto_confirm=args.yes,
                **kwargs
            )

        # Check if user cancelled
        if result is None:
            return 1  # Exit with error code

        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
