"""
Karaoke Creator
Simple script to create karaoke/instrumental versions by removing vocals.
Downloads from YouTube or processes local files.
"""

import sys
import os
import argparse
from pathlib import Path

# Set console encoding for Windows
if sys.platform == 'win32':
    import io
    # Only wrap if not already wrapped
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from youtube_downloader import YouTubeAudioDownloader
from stem_separator import StemSeparator


class KaraokeCreator:
    """Creates karaoke/instrumental versions by removing vocals."""

    def __init__(self, model='htdemucs_ft', device=None, output_dir='karaoke', high_performance=False):
        """
        Initialize the karaoke creator.

        Args:
            model (str): Demucs model to use
            device (str): Device ('cuda', 'cpu', or None for auto-detect)
            output_dir (str): Output directory for karaoke files
            high_performance (bool): Enable high-performance GPU mode
        """
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.high_performance = high_performance

        print("=" * 80)
        print("KARAOKE CREATOR")
        print("=" * 80)
        print(f"Output directory: {self.output_dir}")
        if high_performance:
            print(f"Performance mode: HIGH (maximum GPU utilization)")
        print()

    def create_from_youtube(self, url, filename=None, keep_vocals=True,
                           keep_original=False, output_format='mp3', bitrate=320):
        """
        Download from YouTube and create karaoke version.

        Args:
            url (str): YouTube URL
            filename (str, optional): Custom filename
            keep_vocals (bool): Also save the vocals track
            keep_original (bool): Keep the original downloaded file
            output_format (str): Output format ('mp3' or 'wav')
            bitrate (int): MP3 bitrate in kbps

        Returns:
            dict: Paths to created files
        """
        print("STEP 1: Downloading from YouTube")
        print("-" * 80)

        # Download audio
        downloader = YouTubeAudioDownloader(
            output_dir='downloads',
            format='mp3',
            quality='320'
        )
        audio_file = downloader.download(url, filename)

        print("\nSTEP 2: Creating Karaoke Version")
        print("-" * 80)

        # Create karaoke
        result = self.create_from_file(
            audio_file,
            keep_vocals=keep_vocals,
            output_format=output_format,
            bitrate=bitrate
        )

        # Clean up original download if requested
        if not keep_original:
            print(f"\nRemoving original download: {audio_file}")
            os.remove(audio_file)

        return result

    def create_from_file(self, audio_file, keep_vocals=True,
                        output_format='mp3', bitrate=320):
        """
        Create karaoke version from local audio file.

        Args:
            audio_file (str): Path to audio file
            keep_vocals (bool): Also save the vocals track
            output_format (str): Output format ('mp3' or 'wav')
            bitrate (int): MP3 bitrate in kbps

        Returns:
            dict: Paths to created files
        """
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        print(f"Processing: {audio_path.name}")
        print(f"Model: {self.model}")

        # Initialize separator
        separator = StemSeparator(
            model_name=self.model,
            device=self.device,
            output_dir='separated',  # Temporary output
            high_performance=self.high_performance
        )

        # Separate vocals from everything else
        stems = separator.separate(
            audio_file=audio_file,
            two_stems='vocals',  # Creates vocals + no_vocals
            mp3=(output_format == 'mp3'),
            mp3_bitrate=bitrate
        )

        # Move files to karaoke output directory
        song_name = audio_path.stem
        song_output_dir = self.output_dir / song_name
        song_output_dir.mkdir(parents=True, exist_ok=True)

        result = {}

        # Move instrumental (karaoke) file
        if 'no_vocals' in stems:
            src = Path(stems['no_vocals'])
            ext = 'mp3' if output_format == 'mp3' else 'wav'
            dst = song_output_dir / f"instrumental.{ext}"

            # Copy the file
            import shutil
            shutil.copy2(src, dst)
            result['instrumental'] = dst.as_posix()
            print(f"\nKaraoke version saved: {dst}")

        # Optionally move vocals file
        if keep_vocals and 'vocals' in stems:
            src = Path(stems['vocals'])
            ext = 'mp3' if output_format == 'mp3' else 'wav'
            dst = song_output_dir / f"vocals.{ext}"

            import shutil
            shutil.copy2(src, dst)
            result['vocals'] = dst.as_posix()
            print(f"Vocals saved: {dst}")

        print("\n" + "=" * 80)
        print("KARAOKE CREATION COMPLETE!")
        print("=" * 80)
        print(f"\nOutput directory: {song_output_dir}")

        return result


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Create karaoke/instrumental versions by removing vocals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create karaoke from YouTube URL
  python karaoke.py "https://www.youtube.com/watch?v=VIDEO_ID"

  # Create from local file
  python karaoke.py --local song.mp3

  # Save as WAV format
  python karaoke.py "URL" --format wav

  # Use CPU instead of GPU
  python karaoke.py "URL" --device cpu

  # Don't save vocals track (only instrumental)
  python karaoke.py "URL" --no-vocals

  # Use 6-stem model for better quality
  python karaoke.py "URL" --model htdemucs_6s
        """
    )

    # Input source
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        'url',
        nargs='?',
        help='YouTube URL to download and process'
    )
    input_group.add_argument(
        '--local', '-l',
        metavar='FILE',
        help='Process local audio file'
    )

    # Options
    parser.add_argument(
        '--filename', '-f',
        help='Custom filename for YouTube download (without extension)'
    )
    parser.add_argument(
        '--model', '-m',
        default='htdemucs_ft',
        choices=['htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'mdx_extra'],
        help='Demucs model to use (default: htdemucs_ft)'
    )
    parser.add_argument(
        '--device', '-d',
        choices=['cuda', 'cpu'],
        help='Device to use (default: auto-detect)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='karaoke',
        help='Output directory (default: karaoke)'
    )
    parser.add_argument(
        '--format',
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
        help="Don't save vocals track (only instrumental)"
    )
    parser.add_argument(
        '--keep-original',
        action='store_true',
        help='Keep original downloaded file (for YouTube downloads)'
    )
    parser.add_argument(
        '--high-performance',
        action='store_true',
        help='Enable high-performance GPU mode (faster, uses more VRAM)'
    )

    args = parser.parse_args()

    try:
        creator = KaraokeCreator(
            model=args.model,
            device=args.device,
            output_dir=args.output_dir,
            high_performance=args.high_performance
        )

        if args.local:
            # Process local file
            result = creator.create_from_file(
                audio_file=args.local,
                keep_vocals=not args.no_vocals,
                output_format=args.format,
                bitrate=args.bitrate
            )
        else:
            # Download and process from YouTube
            result = creator.create_from_youtube(
                url=args.url,
                filename=args.filename,
                keep_vocals=not args.no_vocals,
                keep_original=args.keep_original,
                output_format=args.format,
                bitrate=args.bitrate
            )

        print("\nCreated files:")
        for file_type, file_path in result.items():
            print(f"  {file_type}: {file_path}")

        return 0

    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
