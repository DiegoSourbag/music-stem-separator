"""
Music Processor - Coordinator Script
Downloads audio from YouTube and separates it into stems.
Coordinates youtube_downloader.py and stem_separator.py.
"""

import os
import sys
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


class MusicProcessor:
    """Coordinates YouTube download and stem separation."""

    def __init__(self, download_dir='downloads', output_dir='separated',
                 audio_format='mp3', model='htdemucs_ft', device=None, high_performance=False):
        """
        Initialize the music processor.

        Args:
            download_dir (str): Directory for downloaded audio
            output_dir (str): Directory for separated stems
            audio_format (str): Audio format ('mp3' or 'wav')
            model (str): Demucs model to use
            device (str): Device for separation ('cuda', 'cpu', or None)
            high_performance (bool): Enable high-performance GPU mode
        """
        self.download_dir = download_dir
        self.output_dir = output_dir
        self.audio_format = audio_format
        self.model = model
        self.device = device
        self.high_performance = high_performance

        # Initialize components
        self.downloader = YouTubeAudioDownloader(
            output_dir=download_dir,
            format=audio_format,
            quality='320'
        )

        print("=" * 80)
        print("Music Processor Initialized")
        print("=" * 80)

    def process_from_youtube(self, url, filename=None, two_stems=None,
                            output_format='mp3', mp3_bitrate=320,
                            keep_original=True):
        """
        Download audio from YouTube and separate into stems.

        Args:
            url (str): YouTube URL
            filename (str, optional): Custom filename for download
            two_stems (str, optional): Extract only two stems
            output_format (str): Output format for stems ('mp3' or 'wav')
            mp3_bitrate (int): MP3 bitrate for stems
            keep_original (bool): Keep original downloaded file

        Returns:
            tuple: (downloaded_file, stems_dict)
        """
        print("\n" + "=" * 80)
        print("STEP 1: Downloading Audio from YouTube")
        print("=" * 80)

        try:
            # Download audio
            downloaded_file = self.downloader.download(url, filename)

            print("\n" + "=" * 80)
            print("STEP 2: Separating Stems")
            print("=" * 80)

            # Separate stems
            stems = self.process_local_file(
                downloaded_file,
                two_stems=two_stems,
                output_format=output_format,
                mp3_bitrate=mp3_bitrate
            )

            # Optionally remove original download
            if not keep_original:
                print(f"\nRemoving original download: {downloaded_file}")
                os.remove(downloaded_file)

            return downloaded_file, stems

        except Exception as e:
            print(f"\n✗ Processing failed: {str(e)}", file=sys.stderr)
            raise

    def process_local_file(self, audio_file, two_stems=None,
                          output_format='mp3', mp3_bitrate=320):
        """
        Separate an existing audio file into stems.

        Args:
            audio_file (str): Path to audio file
            two_stems (str, optional): Extract only two stems
            output_format (str): Output format for stems ('mp3' or 'wav')
            mp3_bitrate (int): MP3 bitrate for stems

        Returns:
            dict: Dictionary mapping stem names to file paths
        """
        # Initialize separator (lazy loading)
        separator = StemSeparator(
            model_name=self.model,
            device=self.device,
            output_dir=self.output_dir,
            high_performance=self.high_performance
        )

        # Separate stems
        stems = separator.separate(
            audio_file=audio_file,
            two_stems=two_stems,
            mp3=(output_format == 'mp3'),
            mp3_bitrate=mp3_bitrate
        )

        return stems


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Download audio from YouTube and separate into stems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and separate into all stems
  python music_processor.py https://www.youtube.com/watch?v=dQw4w9WgXcQ

  # Download and extract only vocals
  python music_processor.py URL --two-stems vocals

  # Process local file
  python music_processor.py --local song.mp3

  # Use 6-stem model with custom output directory
  python music_processor.py URL --model htdemucs_6s --output-dir my_stems

  # Use CPU instead of GPU
  python music_processor.py URL --device cpu

  # Save stems as WAV
  python music_processor.py URL --stem-format wav

  # Don't keep original download
  python music_processor.py URL --no-keep-original
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
        help='Process local audio file instead of downloading'
    )

    # Download options
    parser.add_argument(
        '--filename', '-f',
        help='Custom filename for download (without extension)'
    )
    parser.add_argument(
        '--download-format',
        choices=['mp3', 'wav'],
        default='mp3',
        help='Format for downloaded audio (default: mp3)'
    )
    parser.add_argument(
        '--download-dir',
        default='downloads',
        help='Directory for downloads (default: downloads)'
    )

    # Separation options
    parser.add_argument(
        '--model', '-m',
        default='htdemucs_ft',
        choices=['htdemucs', 'htdemucs_ft', 'htdemucs_6s', 'mdx_extra'],
        help='Demucs model to use (default: htdemucs_ft)'
    )
    parser.add_argument(
        '--device', '-d',
        choices=['cuda', 'cpu'],
        help='Device to use for separation (default: auto-detect)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='separated',
        help='Directory for separated stems (default: separated)'
    )
    parser.add_argument(
        '--two-stems',
        choices=['vocals', 'drums', 'bass', 'other', 'guitar', 'piano'],
        help='Extract only two stems (selected stem and everything else)'
    )
    parser.add_argument(
        '--stem-format',
        choices=['mp3', 'wav'],
        default='mp3',
        help='Output format for stems (default: mp3)'
    )
    parser.add_argument(
        '--mp3-bitrate',
        type=int,
        default=320,
        help='MP3 bitrate in kbps for stems (default: 320)'
    )

    # Other options
    parser.add_argument(
        '--no-keep-original',
        action='store_true',
        help="Don't keep original downloaded file"
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit'
    )
    parser.add_argument(
        '--high-performance',
        action='store_true',
        help='Enable high-performance GPU mode (faster, uses more VRAM)'
    )

    args = parser.parse_args()

    # Handle list models
    if args.list_models:
        StemSeparator.list_models()
        return 0

    try:
        # Initialize processor
        processor = MusicProcessor(
            download_dir=args.download_dir,
            output_dir=args.output_dir,
            audio_format=args.download_format,
            model=args.model,
            device=args.device,
            high_performance=args.high_performance
        )

        # Process based on input type
        if args.local:
            # Process local file
            print("\n" + "=" * 80)
            print("Processing Local File")
            print("=" * 80)

            stems = processor.process_local_file(
                audio_file=args.local,
                two_stems=args.two_stems,
                output_format=args.stem_format,
                mp3_bitrate=args.mp3_bitrate
            )

            print("\n" + "=" * 80)
            print("PROCESSING COMPLETE")
            print("=" * 80)
            print(f"\nSeparated stems:")
            for stem_name, stem_path in stems.items():
                print(f"  {stem_name}: {stem_path}")

        else:
            # Download and process from YouTube
            downloaded_file, stems = processor.process_from_youtube(
                url=args.url,
                filename=args.filename,
                two_stems=args.two_stems,
                output_format=args.stem_format,
                mp3_bitrate=args.mp3_bitrate,
                keep_original=not args.no_keep_original
            )

            print("\n" + "=" * 80)
            print("PROCESSING COMPLETE")
            print("=" * 80)
            if not args.no_keep_original:
                print(f"\nOriginal file: {downloaded_file}")
            print(f"\nSeparated stems:")
            for stem_name, stem_path in stems.items():
                print(f"  {stem_name}: {stem_path}")

        print("\n✓ All operations completed successfully!")
        return 0

    except Exception as e:
        print(f"\n✗ Processing failed: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
