import torch
import torchaudio
import pyphen
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

from tts_api.core.models import ValidationOptions, PostProcessingOptions
# Import the new centralized VAD function
from tts_api.services.audio_processor import get_speech_timestamps as run_vad_on_waveform

@dataclass
class ValidationResult:
    is_ok: bool
    reason: str = ""

@dataclass
class ValidationContext:
    """A stateful object with lazy properties for efficient validation."""
    original_waveform: torch.Tensor
    sample_rate: int
    text_chunk: str
    validation_params: ValidationOptions
    post_processing_params: PostProcessingOptions
    language: str
    _precomputed_vad_timestamps: Optional[List[Dict]] = None

    _vad_results: Optional[Tuple[List[Dict], torch.Tensor]] = None
    _filtered_speech_segments: Optional[List[Dict]] = None

    def _run_vad_once(self):
        if self._vad_results is not None:
            return

        if self._precomputed_vad_timestamps is not None:
            logging.debug("Using pre-computed VAD timestamps for validation.")
            self._vad_results = (self._precomputed_vad_timestamps, None)
            return

        logging.debug("Running VAD transformation for validation context...")
        # Use the centralized function from audio_processor
        timestamps = run_vad_on_waveform(
            self.original_waveform, self.sample_rate, self.post_processing_params
        )
        self._vad_results = (timestamps, None)

    @property
    def initial_speech_segments(self) -> List[Dict]:
        self._run_vad_once()
        return self._vad_results[0]

    @property
    def filtered_speech_segments(self) -> List[Dict]:
        if self._filtered_speech_segments is None:
            segments = self.initial_speech_segments
            for f in ALL_FILTERS:
                segments = f.filter(segments, self)
            self._filtered_speech_segments = segments
        return self._filtered_speech_segments

class AbstractValidator(ABC):
    @abstractmethod
    def is_valid(self, context: ValidationContext) -> ValidationResult:
        pass

class AbstractFilter(ABC):
    @abstractmethod
    def filter(self, segments: List[Dict], context: ValidationContext) -> List[Dict]:
        pass

def to_dbfs(waveform: torch.Tensor) -> float:
    if waveform.numel() == 0: return -180.0
    rms = torch.sqrt(torch.mean(torch.square(waveform)))
    if rms < 1e-9: return -180.0
    return 20 * torch.log10(rms).item()

# STAGE 1
class SilenceThresholdValidator(AbstractValidator):
    def is_valid(self, context: ValidationContext) -> ValidationResult:
        if context.validation_params.max_silence_dbfs is None: return ValidationResult(is_ok=True)
        dbfs = to_dbfs(context.original_waveform)
        if dbfs < context.validation_params.max_silence_dbfs:
            return ValidationResult(is_ok=False, reason=f"Signal level ({dbfs:.1f} dBFS) is below threshold ({context.validation_params.max_silence_dbfs} dBFS)")
        return ValidationResult(is_ok=True)

# STAGE 2
class LowEnergySegmentFilter(AbstractFilter):
    def filter(self, segments: List[Dict], context: ValidationContext) -> List[Dict]:
        if context.post_processing_params.max_voiced_dynamic_range_db is None or len(segments) < 2:
            return segments
        
        segment_dbfs = [to_dbfs(context.original_waveform[..., s['start']:s['end']]) for s in segments]
        if not segment_dbfs: return []
        
        max_dbfs = max(segment_dbfs)
        db_threshold = max_dbfs - context.post_processing_params.max_voiced_dynamic_range_db
        filtered_segments = [seg for seg, db in zip(segments, segment_dbfs) if db >= db_threshold]
        
        num_removed = len(segments) - len(filtered_segments)
        if num_removed > 0: logging.debug(f"LowEnergyFilter removed {num_removed} mumbled segment(s).")
        return filtered_segments

# STAGE 3
class VoicedDurationValidator(AbstractValidator):
    def is_valid(self, context: ValidationContext) -> ValidationResult:
        params = context.validation_params
        if params.min_voiced_duration_per_syllable is None and params.max_voiced_duration_per_syllable is None:
            return ValidationResult(is_ok=True)

        try:
            dic = pyphen.Pyphen(lang=context.language)
            words = context.text_chunk.split()
            if not words:
                return ValidationResult(is_ok=True) # Cannot validate empty text
            syllable_count = sum(len(dic.inserted(word).split('-')) for word in words)
            if syllable_count == 0:
                syllable_count = 1 # Treat symbol-only text etc. as one syllable
        except Exception as e:
            logging.warning(f"Could not count syllables for lang='{context.language}', skipping VoicedDurationValidator. Error: {e}")
            return ValidationResult(is_ok=True)
            
        # Skip validation for very short text based on syllable count
        if params.min_syllables_for_duration_validation is not None:
            if syllable_count < params.min_syllables_for_duration_validation:
                logging.debug(
                    f"Skipping duration validation for short text "
                    f"(syllables: {syllable_count} < threshold: {params.min_syllables_for_duration_validation})."
                )
                return ValidationResult(is_ok=True)

        total_voiced_samples = sum(s['end'] - s['start'] for s in context.filtered_speech_segments)
        total_voiced_seconds = total_voiced_samples / context.sample_rate
        
        if syllable_count == 0: # Should be unreachable due to above logic
            return ValidationResult(is_ok=True)
            
        ratio = total_voiced_seconds / syllable_count

        if params.min_voiced_duration_per_syllable is not None and ratio < params.min_voiced_duration_per_syllable:
            reason = (
                f"Final voiced duration too short: {ratio:.3f}s/syl "
                f"(min: {params.min_voiced_duration_per_syllable:.3f}s/syl)"
            )
            return ValidationResult(is_ok=False, reason=reason)
        if params.max_voiced_duration_per_syllable is not None and ratio > params.max_voiced_duration_per_syllable:
            reason = (
                f"Final voiced duration too long: {ratio:.3f}s/syl "
                f"(max: {params.max_voiced_duration_per_syllable:.3f}s/syl)"
            )
            return ValidationResult(is_ok=False, reason=reason)
        return ValidationResult(is_ok=True)

class MaxContiguousSilenceValidator(AbstractValidator):
    def is_valid(self, context: ValidationContext) -> ValidationResult:
        if context.validation_params.max_contiguous_silence_s is None or not context.filtered_speech_segments:
            return ValidationResult(is_ok=True)

        last_speech_end = 0
        max_silence_samples = 0
        for seg in context.filtered_speech_segments:
            max_silence_samples = max(max_silence_samples, seg['start'] - last_speech_end)
            last_speech_end = seg['end']
        
        max_s = max_silence_samples / context.sample_rate
        if max_s > context.validation_params.max_contiguous_silence_s:
            return ValidationResult(is_ok=False, reason=f"Longest silent gap ({max_s:.2f}s) exceeds max")
        return ValidationResult(is_ok=True)

# STAGE 4
class ClippingValidator(AbstractValidator):
    def is_valid(self, context: ValidationContext) -> ValidationResult:
        if context.validation_params.max_clipping_percentage is None: return ValidationResult(is_ok=True)

        total_speech_samples, total_clipped_samples = 0, 0
        for seg in context.filtered_speech_segments:
            segment_slice = context.original_waveform[..., seg['start']:seg['end']]
            total_speech_samples += segment_slice.numel()
            total_clipped_samples += torch.sum((segment_slice >= 0.999) | (segment_slice <= -0.999)).item()
        
        if total_speech_samples == 0: return ValidationResult(is_ok=True)
        
        clip_percent = (total_clipped_samples / total_speech_samples) * 100
        if clip_percent > context.validation_params.max_clipping_percentage:
            return ValidationResult(is_ok=False, reason=f"Clipping detected in speech: {clip_percent:.2f}%")
        return ValidationResult(is_ok=True)

def _calculate_spectral_centroid_std_dev(context: ValidationContext) -> Optional[float]:
    """
    Calculates the standard deviation of the spectral centroid across voiced segments.
    Returns the float value or None if it cannot be computed.
    """
    # We need at least two segments to calculate a meaningful standard deviation.
    if len(context.filtered_speech_segments) < 2:
        return None

    try:
        spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=2048, power=2)
        segment_centroids = []
        for seg in context.filtered_speech_segments:
            segment_slice = context.original_waveform[..., seg['start']:seg['end']]
            if segment_slice.numel() < spectrogram_transform.n_fft:
                continue
            
            spec = spectrogram_transform(segment_slice)
            freqs = torch.linspace(0, context.sample_rate / 2, steps=spec.shape[-2]).unsqueeze(-1)
            centroid_per_frame = torch.sum(freqs * spec, dim=-2) / (torch.sum(spec, dim=-2) + 1e-9)
            segment_centroids.append(torch.mean(centroid_per_frame).item())
        
        if len(segment_centroids) < 2:
            return None
            
        return torch.tensor(segment_centroids).std().item()

    except Exception as e:
        logging.warning(f"Spectral Centroid calculation failed with an internal error: {e}", exc_info=True)
        return None

class SpectralCentroidValidator(AbstractValidator):
    def is_valid(self, context: ValidationContext) -> ValidationResult:
        params = context.validation_params
        if params.min_spectral_centroid_std_dev is None:
            return ValidationResult(is_ok=True)
        
        std_dev = _calculate_spectral_centroid_std_dev(context)

        if std_dev is None:
            return ValidationResult(is_ok=True)
            
        if std_dev < params.min_spectral_centroid_std_dev:
            reason = f"Spectral centroid std dev ({std_dev:.1f} Hz) is below threshold ({params.min_spectral_centroid_std_dev} Hz)"
            return ValidationResult(is_ok=False, reason=reason)
        
        return ValidationResult(is_ok=True)


PRE_FILTER_VALIDATORS = [SilenceThresholdValidator()]
ALL_FILTERS = [LowEnergySegmentFilter()]
POST_FILTER_VALIDATORS = [VoicedDurationValidator(), MaxContiguousSilenceValidator()]
QUALITY_VALIDATORS = [ClippingValidator(), SpectralCentroidValidator()]

def run_validation_pipeline(
    waveform: torch.Tensor, sample_rate: int, text_chunk: str,
    validation_params: ValidationOptions, post_processing_params: PostProcessingOptions, language: str,
    vad_speech_timestamps: Optional[List[Dict]] = None
) -> ValidationResult:
    context = ValidationContext(
        waveform, sample_rate, text_chunk, validation_params, post_processing_params,
        language, _precomputed_vad_timestamps=vad_speech_timestamps
    )
    
    for v_list in [PRE_FILTER_VALIDATORS, POST_FILTER_VALIDATORS, QUALITY_VALIDATORS]:
        for validator in v_list:
            result = validator.is_valid(context)
            if not result.is_ok: return result

    return ValidationResult(is_ok=True)