"""
Simple stem separator without Unicode issues
Run this directly to separate the downloaded audio
"""

import sys
import torch
from pathlib import Path

# Set console encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio

def separate_audio(audio_file, output_dir='separated'):
    """Separate audio into stems using Demucs."""

    print("=" * 80)
    print("STEM SEPARATOR")
    print("=" * 80)

    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load model
    print("\nLoading model: htdemucs_ft...")
    model = get_model('htdemucs_ft')
    model.to(device)
    model.eval()
    print(f"Model loaded. Stems: {', '.join(model.sources)}")

    # Create output directory
    audio_path = Path(audio_file)
    song_name = audio_path.stem
    output_path = Path(output_dir) / 'htdemucs_ft' / song_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing: {audio_path.name}")
    print(f"Output: {output_path}")

    # Load audio
    print("\nLoading audio...")
    wav = AudioFile(str(audio_path)).read(
        streams=0,
        samplerate=model.samplerate,
        channels=model.audio_channels
    )

    # Convert to numpy if it's a tensor
    if isinstance(wav, torch.Tensor):
        wav = wav.numpy()

    # Convert to tensor
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()
    wav_tensor = torch.from_numpy(wav).to(device)

    if wav_tensor.dim() == 2:
        wav_tensor = wav_tensor.unsqueeze(0)

    # Separate stems
    print("\nSeparating stems (this may take several minutes)...")
    with torch.no_grad():
        sources = apply_model(
            model,
            wav_tensor,
            device=device,
            shifts=1,
            split=True,
            overlap=0.25,
            progress=True
        )

    # Save stems
    sources = sources * ref.std() + ref.mean()
    sources = sources.cpu()

    print("\nSaving stems...")
    for i, stem_name in enumerate(model.sources):
        stem_file = output_path / f"{stem_name}.mp3"
        print(f"  Saving {stem_name}...", end=' ')
        save_audio(
            sources[0, i],
            str(stem_file),
            samplerate=model.samplerate,
            bitrate=320,
            clip='rescale'
        )
        print(f"Done ({stem_file.name})")

    print("\n" + "=" * 80)
    print("SEPARATION COMPLETE!")
    print("=" * 80)
    print(f"\nOutput directory: {output_path}")
    print(f"Stems created: {len(model.sources)}")

    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python separate_simple.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    separate_audio(audio_file)
