"""BS-RoFormer-SW 6-stem adapter for the existing processing workflows."""

import os
import time
from pathlib import Path

import ffmpeg
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as functional
import yaml
from bs_roformer import DEFAULT_MODEL, ensure_model_assets, get_model_from_config
from bs_roformer.inference import SafeLoaderWithTuple
from ml_collections import ConfigDict

from youtube_downloader import YouTubeAudioDownloader


class BSRoformerSeparator:
    """Separate bass, drums, other, vocals, guitar, and piano with BS-RoFormer."""

    def __init__(self, device=None, output_dir='separated', chunk_size=None,
                 progress_callback=None):
        self.device = torch.device(
            device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
        )
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size or self._recommended_chunk_size()
        self.progress_callback = progress_callback

    def _report_progress(self, percentage, message):
        if self.progress_callback:
            self.progress_callback(percentage, message)

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
        self._report_progress(12, 'Loading BS-RoFormer model...')
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

    def _demix_with_progress(self, config, model, mixture):
        """Chunked overlap-add inference with GUI progress callbacks."""
        chunk_size = config.inference.chunk_size
        overlap = config.inference.num_overlap
        step = chunk_size // overlap
        fade_size = chunk_size // 10
        border = chunk_size - step

        if mixture.shape[1] > 2 * border and border > 0:
            mixture = functional.pad(mixture, (border, border), mode='reflect')

        window = torch.ones(chunk_size, device=self.device)
        window[:fade_size] = torch.linspace(0, 1, fade_size, device=self.device)
        window[-fade_size:] = torch.linspace(1, 0, fade_size, device=self.device)

        instruments = list(config.training.instruments)
        result_shape = (len(instruments),) + tuple(mixture.shape)
        mixture = mixture.to(self.device)
        result = torch.zeros(result_shape, dtype=torch.float32, device=self.device)
        counter = torch.zeros(result_shape, dtype=torch.float32, device=self.device)
        total_length = mixture.shape[1]
        total_chunks = (total_length + step - 1) // step

        autocast_device = 'cuda' if self.device.type == 'cuda' else 'cpu'
        with torch.amp.autocast(autocast_device, enabled=self.device.type == 'cuda'):
            with torch.no_grad():
                for chunk_index, offset in enumerate(range(0, total_length, step), 1):
                    part = mixture[:, offset:offset + chunk_size]
                    original_length = part.shape[-1]
                    if original_length < chunk_size:
                        pad_mode = 'reflect' if original_length > chunk_size // 2 + 1 else 'constant'
                        part = functional.pad(part, (0, chunk_size - original_length), mode=pad_mode)

                    predicted = model(part.unsqueeze(0))[0]
                    usable_length = min(original_length, predicted.shape[-1])
                    chunk_window = window.clone()
                    if offset == 0:
                        chunk_window[:fade_size] = 1
                    if offset + chunk_size >= total_length:
                        chunk_window[-fade_size:] = 1

                    target = slice(offset, offset + usable_length)
                    weights = chunk_window[:usable_length]
                    result[..., target] += predicted[..., :usable_length] * weights
                    counter[..., target] += weights

                    fraction = chunk_index / total_chunks
                    percentage = 20 + round(fraction * 65)
                    message = f'Separating chunk {chunk_index}/{total_chunks}'
                    print(f"\r{message} ({percentage}%)", end='', flush=True)
                    self._report_progress(percentage, message)

        print(flush=True)
        estimated = (result / counter.clamp_min(1e-8)).cpu().float().numpy()
        np.nan_to_num(estimated, copy=False, nan=0.0)
        if mixture.shape[1] > 2 * border and border > 0:
            estimated = estimated[..., border:-border]
        return dict(zip(instruments, estimated))

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
        self._report_progress(5, 'Converting input audio...')
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
            self._report_progress(20, 'Starting BS-RoFormer separation...')
            stems = self._demix_with_progress(config, model, mixture)
            print("Separation complete. Encoding output stems...", flush=True)
            self._report_progress(86, 'Encoding output stems...')

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
                percentage = 86 + round(index / total_stems * 13)
                self._report_progress(percentage, f'Saved {stem_name} ({index}/{total_stems})')

            elapsed = time.time() - started_at
            print(f"\nBS-RoFormer completed in {elapsed:.1f} seconds.", flush=True)
            print(f"Output directory: {song_output_dir}", flush=True)
            self._report_progress(100, 'BS-RoFormer separation complete')
            return output_files
        finally:
            if input_wav.exists():
                input_wav.unlink()
            del model
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()


class BSRoformerProcessor:
    """Download coordinator matching MusicProcessor's public workflow."""

    def __init__(self, download_dir='downloads', output_dir='separated', device=None,
                 progress_callback=None):
        self.downloader = YouTubeAudioDownloader(
            output_dir=download_dir, format='mp3', quality='320'
        )
        self.separator = BSRoformerSeparator(
            device=device, output_dir=output_dir,
            progress_callback=progress_callback
        )

    def process_from_youtube(self, url, filename=None, output_format='mp3',
                             mp3_bitrate=320, keep_original=True):
        self.separator._report_progress(1, 'Downloading audio...')
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
