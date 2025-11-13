"""
Example Usage Script
Demonstrates how to use the Music Stem Separator modules programmatically.
"""

from youtube_downloader import YouTubeAudioDownloader
from stem_separator import StemSeparator
from music_processor import MusicProcessor


def example_1_download_only():
    """Example 1: Download audio from YouTube only."""
    print("\n" + "=" * 80)
    print("Example 1: Download Audio from YouTube")
    print("=" * 80)

    # Initialize downloader
    downloader = YouTubeAudioDownloader(
        output_dir='downloads',
        format='mp3',
        quality='320'
    )

    # Download audio
    url = input("Enter YouTube URL: ")
    audio_file = downloader.download(url)

    print(f"\nDownloaded: {audio_file}")


def example_2_separate_local():
    """Example 2: Separate stems from a local audio file."""
    print("\n" + "=" * 80)
    print("Example 2: Separate Stems from Local File")
    print("=" * 80)

    # Initialize separator
    separator = StemSeparator(
        model_name='htdemucs_ft',  # Best overall model
        device=None,  # Auto-detect GPU/CPU
        output_dir='separated'
    )

    # Separate stems
    audio_file = input("Enter path to audio file: ")
    stems = separator.separate(
        audio_file=audio_file,
        two_stems=None,  # Separate all stems
        mp3=True,
        mp3_bitrate=320
    )

    print("\nSeparated stems:")
    for stem_name, stem_path in stems.items():
        print(f"  {stem_name}: {stem_path}")


def example_3_download_and_separate():
    """Example 3: Download from YouTube and separate (all-in-one)."""
    print("\n" + "=" * 80)
    print("Example 3: Download and Separate (All-in-One)")
    print("=" * 80)

    # Initialize processor
    processor = MusicProcessor(
        download_dir='downloads',
        output_dir='separated',
        audio_format='mp3',
        model='htdemucs_ft',
        device=None  # Auto-detect
    )

    # Process YouTube URL
    url = input("Enter YouTube URL: ")
    audio_file, stems = processor.process_from_youtube(
        url=url,
        two_stems=None,  # All stems
        output_format='mp3',
        mp3_bitrate=320,
        keep_original=True
    )

    print(f"\nOriginal file: {audio_file}")
    print("\nSeparated stems:")
    for stem_name, stem_path in stems.items():
        print(f"  {stem_name}: {stem_path}")


def example_4_extract_vocals_only():
    """Example 4: Extract only vocals from YouTube."""
    print("\n" + "=" * 80)
    print("Example 4: Extract Vocals Only")
    print("=" * 80)

    processor = MusicProcessor(
        model='htdemucs_ft',
        device=None
    )

    url = input("Enter YouTube URL: ")
    audio_file, stems = processor.process_from_youtube(
        url=url,
        two_stems='vocals',  # Extract vocals + no_vocals
        output_format='mp3',
        keep_original=False  # Delete original download
    )

    print("\nExtracted stems:")
    for stem_name, stem_path in stems.items():
        print(f"  {stem_name}: {stem_path}")


def example_5_6stem_separation():
    """Example 5: Use 6-stem model for detailed separation."""
    print("\n" + "=" * 80)
    print("Example 5: 6-Stem Separation (Detailed)")
    print("=" * 80)

    processor = MusicProcessor(
        model='htdemucs_6s',  # 6-stem model
        device=None
    )

    url = input("Enter YouTube URL: ")
    audio_file, stems = processor.process_from_youtube(
        url=url,
        output_format='wav',  # High-quality WAV output
        keep_original=True
    )

    print(f"\nOriginal file: {audio_file}")
    print("\nSeparated stems (6-stem model):")
    for stem_name, stem_path in stems.items():
        print(f"  {stem_name}: {stem_path}")


def main():
    """Main menu for examples."""
    print("=" * 80)
    print("Music Stem Separator - Example Usage")
    print("=" * 80)
    print("\nChoose an example to run:")
    print("  1. Download audio from YouTube only")
    print("  2. Separate stems from local audio file")
    print("  3. Download and separate (all-in-one)")
    print("  4. Extract vocals only from YouTube")
    print("  5. 6-stem separation for detailed instruments")
    print("  0. Exit")

    choice = input("\nEnter choice (0-5): ").strip()

    examples = {
        '1': example_1_download_only,
        '2': example_2_separate_local,
        '3': example_3_download_and_separate,
        '4': example_4_extract_vocals_only,
        '5': example_5_6stem_separation,
    }

    if choice == '0':
        print("Exiting...")
        return

    example_func = examples.get(choice)
    if example_func:
        try:
            example_func()
            print("\n✓ Example completed successfully!")
        except Exception as e:
            print(f"\n✗ Error: {str(e)}")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
