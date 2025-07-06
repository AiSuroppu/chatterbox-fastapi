import ffmpeg
import logging
import io
import torch
import torchaudio
import silero_vad
from typing import List, Dict, Optional

from tts_api.core.models import PostProcessingOptions, ExportFormat, WordSegment

VAD_INTERNAL_SAMPLE_RATE = 16000 # The fixed rate Silero VAD operates on.

# Global caches for VAD model and resamplers for efficiency
_VAD_MODEL_CACHE = None
_RESAMPLER_CACHE = {}


def _get_vad_model():
    global _VAD_MODEL_CACHE
    if _VAD_MODEL_CACHE is None:
        logging.info("Loading Silero-VAD model into cache...")
        # The 'use_onnx=True' flag is recommended for speed.
        model = silero_vad.load_silero_vad(onnx=True)
        _VAD_MODEL_CACHE = model
        logging.info("Silero-VAD model loaded successfully.")
    return _VAD_MODEL_CACHE

def _get_resampler(from_rate: int, to_rate: int):
    """Gets a cached or new torchaudio resampler."""
    if (from_rate, to_rate) not in _RESAMPLER_CACHE:
        _RESAMPLER_CACHE[(from_rate, to_rate)] = torchaudio.transforms.Resample(
            orig_freq=from_rate, new_freq=to_rate
        )
    return _RESAMPLER_CACHE[(from_rate, to_rate)]

def _merge_speech_timestamps(
    vad_timestamps: List[Dict[str, int]],
    word_segments: Optional[List[WordSegment]],
    sample_rate: int
) -> List[Dict[str, int]]:
    """
    Merges VAD timestamps with word alignment timestamps to create a consolidated
    list of speech segments, preventing words from being cut by the VAD.
    """
    if not word_segments:
        return vad_timestamps
    
    # Convert word segments (in seconds) to the same sample-based format as VAD timestamps
    word_timestamps = []
    for word in word_segments:
        if word.start is not None and word.end is not None:
            word_timestamps.append({
                'start': int(word.start * sample_rate),
                'end': int(word.end * sample_rate)
            })

    # Combine and sort all timestamp segments
    all_timestamps = sorted(vad_timestamps + word_timestamps, key=lambda x: x['start'])

    if not all_timestamps:
        return []
    
    # Merge overlapping or adjacent segments
    merged = [all_timestamps[0]]
    for current in all_timestamps[1:]:
        last = merged[-1]
        # If the current segment overlaps or is adjacent to the last one, merge them
        if current['start'] <= last['end']:
            last['end'] = max(last['end'], current['end'])
        else:
            merged.append(current)
            
    return merged

def get_speech_timestamps(
    waveform: torch.Tensor,
    sample_rate: int,
    options: PostProcessingOptions
) -> List[Dict[str, int]]:
    """
    Runs VAD on a waveform and returns speech timestamps.

    The returned timestamps are in samples, scaled to the original `sample_rate`.
    """
    model = _get_vad_model()
    mono_waveform = torch.mean(waveform, dim=0) if waveform.ndim > 1 else waveform

    if sample_rate != VAD_INTERNAL_SAMPLE_RATE:
        resampler = _get_resampler(sample_rate, VAD_INTERNAL_SAMPLE_RATE)
        vad_input_waveform = resampler(mono_waveform)
    else:
        vad_input_waveform = mono_waveform

    timestamps_16k = silero_vad.get_speech_timestamps(
        vad_input_waveform,
        model,
        sampling_rate=VAD_INTERNAL_SAMPLE_RATE,
        threshold=options.vad_threshold,
        min_speech_duration_ms=options.vad_min_speech_ms,
        min_silence_duration_ms=options.vad_min_silence_ms,
        speech_pad_ms=options.vad_speech_pad_ms
    )

    if not timestamps_16k:
        return []

    # Scale timestamps back to the original sample rate
    scale_factor = sample_rate / VAD_INTERNAL_SAMPLE_RATE
    original_rate_timestamps = [
        {'start': int(ts['start'] * scale_factor), 'end': int(ts['end'] * scale_factor)}
        for ts in timestamps_16k
    ]
    return original_rate_timestamps

def calculate_speech_ratio_from_timestamps(
    speech_timestamps: List[Dict[str, int]],
    total_samples: int
) -> float:
    """Calculates speech ratio from pre-computed VAD timestamps."""
    if not speech_timestamps or total_samples == 0:
        return 0.0

    total_speech_samples = sum(s['end'] - s['start'] for s in speech_timestamps)
    return total_speech_samples / total_samples

def get_speech_ratio(
    waveform: torch.Tensor,
    sample_rate: int,
    options: PostProcessingOptions
) -> float:
    """
    Calculates the ratio of speech to total duration in a waveform using VAD.
    """
    timestamps = get_speech_timestamps(waveform, sample_rate, options)
    return calculate_speech_ratio_from_timestamps(timestamps, waveform.shape[-1])


def apply_vad_denoising(
    waveform: torch.Tensor,
    sample_rate: int,
    options: PostProcessingOptions,
    word_segments: Optional[List[WordSegment]] = None
) -> torch.Tensor:
    """
    Applies VAD to a waveform tensor to remove noise in silent segments.
    This version is alignment-aware, merging VAD results with word timestamps
    to prevent accidentally cutting out words.
    """
    # 1. Get speech timestamps from VAD as a baseline
    vad_timestamps = get_speech_timestamps(waveform, sample_rate, options)
    
    # 2. Merge with word alignment data to get a consolidated view of speech
    consolidated_timestamps = _merge_speech_timestamps(vad_timestamps, word_segments, sample_rate)

    # If no speech is detected by either system, return pure silence.
    if not consolidated_timestamps:
        return torch.zeros_like(waveform)
    
    output_waveform = torch.zeros_like(waveform)
    fade_samples = int((options.vad_fade_ms / 1000) * sample_rate)

    for segment in consolidated_timestamps:
        # Timestamps are already scaled for the original waveform
        start_sample = segment['start']
        end_sample = segment['end']
        
        # Clamp segments to be within the waveform bounds
        start_sample = max(0, start_sample)
        end_sample = min(waveform.shape[-1], end_sample)
        
        if start_sample >= end_sample:
            continue
            
        segment_to_copy = waveform[..., start_sample:end_sample].clone()
        
        segment_length = segment_to_copy.shape[-1]
        current_fade_len = min(fade_samples, segment_length // 2)

        if current_fade_len > 0:
            fade_in = torch.linspace(0.0, 1.0, current_fade_len, device=waveform.device)
            fade_out = torch.linspace(1.0, 0.0, current_fade_len, device=waveform.device)
            
            segment_to_copy[..., :current_fade_len] *= fade_in
            segment_to_copy[..., -current_fade_len:] *= fade_out

        output_waveform[..., start_sample:end_sample] = segment_to_copy

    return output_waveform

def post_process_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    options: PostProcessingOptions,
    word_segments: Optional[List[WordSegment]] = None
) -> io.BytesIO:
    """
    Applies requested post-processing to an in-memory audio waveform and returns
    the final audio as an in-memory BytesIO object.
    """
    processed_waveform = waveform.cpu()

    # 1. Apply VAD Denoising in-memory (if requested)
    if options.denoise_with_vad:
        logging.info("Applying VAD denoiser (alignment-aware)...")
        processed_waveform = apply_vad_denoising(
            processed_waveform, 
            sample_rate, 
            options,
            word_segments
        )
        logging.info("VAD denoiser applied successfully.")

    # Convert tensor to raw audio bytes for ffmpeg's stdin
    # Ensure waveform is 2D: (channels, samples)
    if processed_waveform.ndim == 1:
        processed_waveform = processed_waveform.unsqueeze(0)
    
    # Handle empty waveform after processing
    if processed_waveform.numel() == 0:
        logging.warning("Waveform is empty after processing, returning empty audio.")
        return io.BytesIO()

    # ffmpeg expects interleaved PCM data
    raw_audio_bytes = processed_waveform.numpy().tobytes()

    # 2. Set up the ffmpeg pipeline
    input_stream = ffmpeg.input(
        'pipe:', 
        format='f32le', # 32-bit floating-point, little-endian
        ac=processed_waveform.shape[0], # Number of channels
        ar=str(sample_rate)
    )

    # 3. Apply Normalization if requested
    if options.normalize_audio:
        logging.info("Applying ffmpeg normalization...")
        if options.normalize_method == "ebu":
            input_stream = input_stream.filter('loudnorm', i=options.normalize_level, tp=-2.0, lra=7)
        else: # peak
            input_stream = input_stream.filter('dynaudnorm')

    # 4. Define output format and arguments
    output_args = {'ar': str(sample_rate)}
    if options.export_format == ExportFormat.MP3:
        output_format = 'mp3'
        # Use the configurable constant bitrate to ensure timing is preserved.
        output_args['audio_bitrate'] = options.mp3_bitrate
    elif options.export_format == ExportFormat.FLAC:
        output_format = 'flac'
    else: # WAV
        output_format = 'wav'
        # For WAV, it's often best to specify the codec to avoid metadata issues
        output_args['acodec'] = 'pcm_s16le' # 16-bit signed PCM

    # 5. Execute ffmpeg, piping data in and out
    try:
        logging.info(f"Converting to {output_format.upper()}...")
        out_bytes, err = (
            input_stream.output('pipe:', format=output_format, **output_args)
            .run(input=raw_audio_bytes, capture_stdout=True, capture_stderr=True, quiet=True)
        )
        logging.info("Audio post-processing complete.")
        return io.BytesIO(out_bytes)
    except ffmpeg.Error as e:
        # The error from ffmpeg is in stderr
        logging.error(f"ffmpeg pipeline failed: {e.stderr.decode()}")
        raise RuntimeError("Failed to process audio with ffmpeg.")