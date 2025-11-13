"""
YouTube Audio Downloader
Downloads audio from YouTube URLs in MP3 or WAV format.
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

import yt_dlp


class YouTubeAudioDownloader:
    """Downloads audio from YouTube URLs."""

    def __init__(self, output_dir="downloads", format="mp3", quality="320"):
        """
        Initialize the downloader.

        Args:
            output_dir (str): Directory to save downloaded files
            format (str): Audio format ('mp3' or 'wav')
            quality (str): Audio quality/bitrate for MP3 (e.g., '320', '192')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.format = format.lower()
        self.quality = quality

    def download(self, url, filename=None):
        """
        Download audio from YouTube URL.

        Args:
            url (str): YouTube URL
            filename (str, optional): Custom filename (without extension)

        Returns:
            str: Path to downloaded file
        """
        # Configure output template
        if filename:
            output_template = str(self.output_dir / f"{filename}.%(ext)s")
        else:
            output_template = str(self.output_dir / "%(title)s.%(ext)s")

        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_template,
            'quiet': False,
            'no_warnings': False,
            'extract_audio': True,
        }

        # Add format-specific options
        if self.format == 'mp3':
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': self.quality,
            }]
        elif self.format == 'wav':
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }]
        else:
            raise ValueError(f"Unsupported format: {self.format}. Use 'mp3' or 'wav'.")

        try:
            print(f"Downloading audio from: {url}")
            print(f"Format: {self.format.upper()}")
            print(f"Output directory: {self.output_dir}")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

                # Get the actual filename that was created
                if filename:
                    expected_file = self.output_dir / f"{filename}.{self.format}"
                else:
                    # Try to get the exact filename yt-dlp created
                    # Use yt-dlp's prepare_filename with the audio extension
                    prepared = ydl.prepare_filename(info)
                    # Replace the extension with our target format
                    import os
                    base = os.path.splitext(prepared)[0]
                    expected_file = self.output_dir / f"{os.path.basename(base)}.{self.format}"

                # Verify the file exists, if not try to find it with glob
                if not expected_file.exists():
                    # Try to find the file by pattern matching
                    title = info.get('title', 'audio')
                    # Search for any file starting with first few words of title
                    first_words = ' '.join(title.split()[:3])
                    import glob
                    pattern = str(self.output_dir / f"{first_words}*.{self.format}")
                    matches = glob.glob(pattern)
                    if matches:
                        expected_file = Path(matches[0])
                    else:
                        # Fall back to safe title
                        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
                        expected_file = self.output_dir / f"{safe_title}.{self.format}"

                downloaded_file = expected_file

                print(f"\n✓ Download completed!")
                print(f"  File: {downloaded_file}")
                print(f"  Title: {info.get('title', 'Unknown')}")
                print(f"  Duration: {info.get('duration', 0)} seconds")

                return downloaded_file.as_posix()

        except Exception as e:
            print(f"\n✗ Error downloading audio: {str(e)}", file=sys.stderr)
            raise


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Download audio from YouTube URLs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python youtube_downloader.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
  python youtube_downloader.py https://youtu.be/dQw4w9WgXcQ --format wav
  python youtube_downloader.py URL --output my_song --quality 192
        """
    )

    parser.add_argument(
        'url',
        help='YouTube URL to download'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output filename (without extension)',
        default=None
    )
    parser.add_argument(
        '--format', '-f',
        choices=['mp3', 'wav'],
        default='mp3',
        help='Audio format (default: mp3)'
    )
    parser.add_argument(
        '--quality', '-q',
        default='320',
        help='Audio quality/bitrate for MP3 (default: 320)'
    )
    parser.add_argument(
        '--output-dir', '-d',
        default='downloads',
        help='Output directory (default: downloads)'
    )

    args = parser.parse_args()

    try:
        downloader = YouTubeAudioDownloader(
            output_dir=args.output_dir,
            format=args.format,
            quality=args.quality
        )

        downloaded_file = downloader.download(args.url, args.output)
        return 0

    except Exception as e:
        print(f"\nFailed to download audio: {str(e)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
