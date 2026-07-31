"""BS-RoFormer-SW 6-stem adapter for the existing processing workflows."""

import os
import time
from pathlib import Path

import ffmpeg
import numpy as np
import soundfile as sf
import torch
import yaml
from bs_roformer import DEFAULT_MODEL, demix_track, ensure_model_assets, get_model_from_config
from bs_roformer.inference import SafeLoaderWithTuple
from ml_collections import ConfigDict

from youtube_downloader import YouTubeAudioDownloader


class BSRoformerSeparator:
    """Separate bass, drums, other, vocals, guitar, and piano with BS-RoFormer."""

    def __init__(self, device=None, output_dir='separated', chunk_size=None):
        self.device = torch.device(
            device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size or self._recommended_chunk_size()

    def _recommended_chunk_size(self):
        if self.device.type != 'cuda':
            return 176128
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        # The model's STFT hop is 512 samples, so chunks must stay on that grid.
        return 176128 if memory_gb <= 8 else 352256

    def _load_model(self):
        print("\n" + "=" * 80, flush=True)
        print("BS-ROFORMER-SW 6-STEM SEPARATION", flush=True)
        print("=" * 80, flush=True)
        print(f"Device: {self.device}", flush=True)
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"Chunk size: {self.chunk_size} samples", flush=True)
        print("Loading model and verifying cached weights...", flush=True)
        model_path, config_path = ensure_model_assets(DEFAULT_MODEL)
        with open(config_path, encoding='utf-8') as config_file:
            config = ConfigDict(yaml.load(config_file, Loader=SafeLoaderWithTuple))
        config.inference.chunk_size = self.chunk_size

        model = get_model_from_config('bs_roformer', config)
        state = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        model.to(self.device).eval()
        print("Model loaded successfully.", flush=True)
        return model, config

    def separate(self, audio_file, output_format='mp3', bitrate=320):
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        song_output_dir = self.output_dir / 'bs_roformer_sw_6s' / audio_path.stem
        song_output_dir.mkdir(parents=True, exist_ok=True)
        input_wav = song_output_dir / '_input.wav'

        started_at = time.time()
        print(f"\nPreparing audio: {audio_path.name}", flush=True)
        print("Converting input to stereo 44.1 kHz WAV...", flush=True)
        ffmpeg.output(
            ffmpeg.input(str(audio_path)).audio,
            str(input_wav), ac=2, ar=44100, acodec='pcm_f32le'
        ).overwrite_output().run(quiet=True)

        model = None
        try:
            audio, samplerate = sf.read(input_wav, dtype='float32', always_2d=True)
            mixture = torch.from_numpy(np.ascontiguousarray(audio.T))
            model, config = self._load_model()
            print("Separating audio in overlapping chunks...", flush=True)
            stems, _ = demix_track(config, model, mixture, self.device, None)
            print("Separation complete. Encoding output stems...", flush=True)

            output_files = {}
            total_stems = len(stems)
            for index, (stem_name, stem_audio) in enumerate(stems.items(), 1):
                print(f"  [{index}/{total_stems}] Saving {stem_name}...", end=' ', flush=True)
                wav_path = song_output_dir / f'{stem_name}.wav'
                sf.write(wav_path, stem_audio.T, samplerate, subtype='FLOAT')
                if output_format == 'mp3':
                    output_path = song_output_dir / f'{stem_name}.mp3'
                    ffmpeg.output(
                        ffmpeg.input(str(wav_path)).audio,
                        str(output_path), audio_bitrate=f'{bitrate}k'
                    ).overwrite_output().run(quiet=True)
                    wav_path.unlink()
                else:
                    output_path = wav_path
                output_files[stem_name] = output_path.as_posix()
                print(f"done ({output_path.name})", flush=True)

            elapsed = time.time() - started_at
            print(f"\nBS-RoFormer completed in {elapsed:.1f} seconds.", flush=True)
            print(f"Output directory: {song_output_dir}", flush=True)
            return output_files
        finally:
            if input_wav.exists():
                input_wav.unlink()
            del model
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()


class BSRoformerProcessor:
    """Download coordinator matching MusicProcessor's public workflow."""

    def __init__(self, download_dir='downloads', output_dir='separated', device=None):
        self.downloader = YouTubeAudioDownloader(
            output_dir=download_dir, format='mp3', quality='320'
        )
        self.separator = BSRoformerSeparator(device=device, output_dir=output_dir)

    def process_from_youtube(self, url, filename=None, output_format='mp3',
                             mp3_bitrate=320, keep_original=True):
        downloaded_file = self.downloader.download(url, filename)
        try:
            stems = self.separator.separate(
                downloaded_file, output_format=output_format, bitrate=mp3_bitrate
            )
            return downloaded_file, stems
        finally:
            if not keep_original and Path(downloaded_file).exists():
                os.remove(downloaded_file)

    def process_local_file(self, audio_file, output_format='mp3', mp3_bitrate=320):
        return self.separator.separate(
            audio_file, output_format=output_format, bitrate=mp3_bitrate
        )
