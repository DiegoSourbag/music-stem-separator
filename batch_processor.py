"""
Batch Processor
Process multiple YouTube URLs from a text file.
Creates karaoke versions or full stem separations.
"""

import sys
import os
import argparse
import time
from pathlib import Path
from datetime import datetime

# Set console encoding for Windows
if sys.platform == 'win32':
    import io
    # Only wrap if not already wrapped
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if not isinstance(sys.stderr, io.TextIOWrapper) or sys.stderr.encoding != 'utf-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from karaoke import KaraokeCreator
from music_processor import MusicProcessor


class BatchProcessor:
    """Process multiple songs from a URL list file."""

    def __init__(self, mode='karaoke', model='htdemucs_ft', device=None,
                 output_dir=None, output_format='mp3', bitrate=320, high_performance=False):
        """
        Initialize the batch processor.

        Args:
            mode (str): Processing mode ('karaoke' or 'full')
            model (str): Demucs model to use
            device (str): Device ('cuda', 'cpu', or None)
            output_dir (str): Output directory
            output_format (str): Output format ('mp3' or 'wav')
            bitrate (int): MP3 bitrate in kbps
            high_performance (bool): Enable high-performance GPU mode
        """
        self.mode = mode
        self.model = model
        self.device = device
        self.output_format = output_format
        self.bitrate = bitrate
        self.high_performance = high_performance

        # Set default output directory based on mode
        if output_dir is None:
            self.output_dir = 'karaoke' if mode == 'karaoke' else 'separated'
        else:
            self.output_dir = output_dir

        # Initialize processor based on mode
        if mode == 'karaoke':
            self.processor = KaraokeCreator(
                model=model,
                device=device,
                output_dir=self.output_dir,
                high_performance=high_performance
            )
        else:
            self.processor = MusicProcessor(
                model=model,
                device=device,
                output_dir=self.output_dir,
                high_performance=high_performance
            )

        # Statistics
        self.total = 0
        self.successful = 0
        self.failed = 0
        self.errors = []

    def load_urls(self, file_path):
        """
        Load URLs from a text file.

        Args:
            file_path (str): Path to text file with URLs

        Returns:
            list: List of URLs
        """
        urls = []
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"URL list file not found: {file_path}")

        print(f"Loading URLs from: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # Basic URL validation
                if line.startswith('http://') or line.startswith('https://'):
                    urls.append(line)
                else:
                    print(f"Warning: Skipping invalid URL on line {line_num}: {line}")

        print(f"Loaded {len(urls)} valid URLs\n")
        return urls

    def process_urls(self, urls, keep_vocals=True, keep_original=False,
                    delay=0, continue_on_error=True):
        """
        Process a list of URLs.

        Args:
            urls (list): List of YouTube URLs
            keep_vocals (bool): Keep vocals track (karaoke mode only)
            keep_original (bool): Keep original downloads
            delay (int): Delay between downloads in seconds
            continue_on_error (bool): Continue processing if a URL fails

        Returns:
            dict: Processing statistics
        """
        self.total = len(urls)
        self.successful = 0
        self.failed = 0
        self.errors = []

        start_time = time.time()

        print("=" * 80)
        print(f"BATCH PROCESSING: {self.total} URLs")
        print(f"Mode: {self.mode.upper()}")
        print(f"Model: {self.model}")
        print("=" * 80)
        print()

        for idx, url in enumerate(urls, 1):
            print("\n" + "=" * 80)
            print(f"Processing {idx}/{self.total}: {url}")
            print("=" * 80)

            try:
                if self.mode == 'karaoke':
                    # Karaoke mode
                    self.processor.create_from_youtube(
                        url=url,
                        keep_vocals=keep_vocals,
                        keep_original=keep_original,
                        output_format=self.output_format,
                        bitrate=self.bitrate
                    )
                else:
                    # Full stem separation mode
                    self.processor.process_from_youtube(
                        url=url,
                        output_format=self.output_format,
                        mp3_bitrate=self.bitrate,
                        keep_original=keep_original
                    )

                self.successful += 1
                print(f"\n[{idx}/{self.total}] SUCCESS")

            except Exception as e:
                self.failed += 1
                error_msg = f"URL {idx}: {url}\nError: {str(e)}"
                self.errors.append(error_msg)

                print(f"\n[{idx}/{self.total}] FAILED: {str(e)}", file=sys.stderr)

                if not continue_on_error:
                    print("\nStopping batch processing due to error.")
                    break

            # Delay between downloads (if specified)
            if delay > 0 and idx < self.total:
                print(f"\nWaiting {delay} seconds before next download...")
                time.sleep(delay)

        # Print summary
        elapsed = time.time() - start_time
        self._print_summary(elapsed)

        return {
            'total': self.total,
            'successful': self.successful,
            'failed': self.failed,
            'errors': self.errors,
            'elapsed': elapsed
        }

    def _print_summary(self, elapsed):
        """Print processing summary."""
        print("\n" + "=" * 80)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 80)
        print(f"\nTotal URLs: {self.total}")
        print(f"Successful: {self.successful}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {(self.successful/self.total*100):.1f}%")
        print(f"Elapsed Time: {elapsed:.1f} seconds")
        print(f"Average Time per Song: {(elapsed/self.total):.1f} seconds")

        if self.errors:
            print("\n" + "-" * 80)
            print("ERRORS:")
            print("-" * 80)
            for error in self.errors:
                print(error)
                print("-" * 40)

    def save_error_log(self, log_file='batch_errors.log'):
        """Save errors to a log file."""
        if not self.errors:
            return

        log_path = Path(log_file)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Batch Processing Errors - {timestamp}\n")
            f.write(f"{'=' * 80}\n\n")
            for error in self.errors:
                f.write(f"{error}\n\n")

        print(f"\nErrors saved to: {log_path}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Process multiple YouTube URLs from a text file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
URL List File Format:
  - One URL per line
  - Lines starting with # are treated as comments
  - Empty lines are ignored

Examples:
  # Create karaoke versions from URL list
  python batch_processor.py urls.txt

  # Full stem separation for all URLs
  python batch_processor.py urls.txt --mode full

  # Use 6-stem model and save as WAV
  python batch_processor.py urls.txt --model htdemucs_6s --format wav

  # Add 5-second delay between downloads
  python batch_processor.py urls.txt --delay 5

  # Use CPU mode
  python batch_processor.py urls.txt --device cpu

  # Keep original downloads
  python batch_processor.py urls.txt --keep-original

Sample urls.txt file:
  # My favorite songs
  https://www.youtube.com/watch?v=dQw4w9WgXcQ
  https://www.youtube.com/watch?v=VIDEO_ID2

  # More songs
  https://youtu.be/SHORT_ID
        """
    )

    parser.add_argument(
        'url_file',
        help='Text file containing YouTube URLs (one per line)'
    )
    parser.add_argument(
        '--mode',
        choices=['karaoke', 'full'],
        default='karaoke',
        help='Processing mode: karaoke (vocals/instrumental) or full (all stems). Default: karaoke'
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
        help='Output directory (default: karaoke/ or separated/ based on mode)'
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
        help="Don't save vocals track (karaoke mode only)"
    )
    parser.add_argument(
        '--keep-original',
        action='store_true',
        help='Keep original downloaded files'
    )
    parser.add_argument(
        '--delay',
        type=int,
        default=0,
        help='Delay between downloads in seconds (default: 0)'
    )
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Stop processing if any URL fails (default: continue)'
    )
    parser.add_argument(
        '--save-errors',
        action='store_true',
        help='Save errors to batch_errors.log file'
    )
    parser.add_argument(
        '--high-performance',
        action='store_true',
        help='Enable high-performance GPU mode (faster, uses more VRAM)'
    )

    args = parser.parse_args()

    try:
        # Initialize batch processor
        processor = BatchProcessor(
            mode=args.mode,
            model=args.model,
            device=args.device,
            output_dir=args.output_dir,
            output_format=args.format,
            bitrate=args.bitrate,
            high_performance=args.high_performance
        )

        # Load URLs
        urls = processor.load_urls(args.url_file)

        if not urls:
            print("Error: No valid URLs found in file", file=sys.stderr)
            return 1

        # Process URLs
        stats = processor.process_urls(
            urls=urls,
            keep_vocals=not args.no_vocals,
            keep_original=args.keep_original,
            delay=args.delay,
            continue_on_error=not args.stop_on_error
        )

        # Save error log if requested
        if args.save_errors and processor.errors:
            processor.save_error_log()

        # Return non-zero exit code if any failures
        return 0 if stats['failed'] == 0 else 1

    except Exception as e:
        print(f"\nBatch processing failed: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
