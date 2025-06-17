import re
import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import torch
import torchaudio

from tts_api.core.models import ValidationOptions, PostProcessingOptions
from tts_api.services.audio_processor import get_speech_timestamps as run_vad_on_waveform
from tts_api.utils.pyphen_cache import get_pyphen

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
    _syllable_count: Optional[int] = None

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

    @property
    def syllable_count(self) -> int:
        """
        Calculates and caches the syllable count for the text_chunk.
        Returns 0 if syllable counting fails or text is empty.
        """
        if self._syllable_count is not None:
            return self._syllable_count

        try:
            # Use our globally cached factory to get the pyphen object
            dic = get_pyphen(lang=self.language)
            words = self.text_chunk.split()
            if not words:
                self._syllable_count = 0
                return 0

            count = sum(len(dic.inserted(word).split('-')) for word in words)
            # Treat symbol-only text etc. as one syllable
            self._syllable_count = count if count > 0 else 1
            return self._syllable_count
        except Exception as e:
            # If counting fails for any reason, log a warning and return 0
            # to prevent validation from crashing.
            logging.warning(f"Syllable counting failed for lang='{self.language}': {e}. Returning 0 syllables.")
            self._syllable_count = 0
            return 0

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
    
    def _calculate_lnv(self, text: str) -> Optional[float]:
        """Calculates the Log-Normalized Variety (LNV) of a text string."""
        seg_len = len(text)
        if seg_len <= 1:
            return None
        
        unique_chars = len(set(text))
        return unique_chars / math.log(seg_len)

    def _calculate_dynamic_duration_budgets(self, context: ValidationContext) -> Tuple[int, float, float]:
        """
        Calculates dynamic min/max duration budgets by analyzing each word's complexity.

        Returns:
            A tuple of (total_syllables, min_duration_budget, max_duration_budget).
        """
        params = context.validation_params
        dic = get_pyphen(lang=context.language)
        
        words = context.text_chunk.split()
        if not words:
            return 0, 0.0, 0.0

        total_syllables = 0
        max_duration_budget = 0.0

        for word in words:
            # 1. Calculate syllables on the original word.
            try:
                # Use pyphen to count syllables, fallback to 1 for errors/non-words.
                syllables = len(dic.inserted(word).split('-'))
                syllables = syllables if syllables > 0 else 1
            except Exception:
                syllables = 1
            total_syllables += syllables

            # 2. Clean word for LNV analysis.
            cleaned_word = re.sub(r'^\W+|\W+$', '', word)
            if not cleaned_word:
                # Budget punctuation using the standard rate.
                max_duration_budget += syllables * params.max_voiced_duration_per_syllable
                continue
            
            # 3. Check if word is long enough for low-complexity analysis.
            if len(cleaned_word) < params.min_word_len_for_low_complexity_analysis:
                # Word is too short, treat as normal complexity.
                max_duration_budget += syllables * params.max_voiced_duration_per_syllable
                continue

            # 4. Assess complexity of eligible words and allocate budget.
            lnv = self._calculate_lnv(cleaned_word)
            is_low_complexity = (lnv is not None and lnv < params.low_complexity_log_variety_threshold)

            if is_low_complexity:
                # Use the relaxed rate for stretched words/onomatopoeia.
                max_duration_budget += syllables * params.low_complexity_max_duration_per_syllable
                logging.debug(f"Word '{cleaned_word}' (LNV: {lnv:.2f}) budgeted with relaxed rate.")
            else:
                # Use the standard, stricter rate for normal words.
                max_duration_budget += syllables * params.max_voiced_duration_per_syllable
        
        min_duration_budget = total_syllables * params.min_voiced_duration_per_syllable if params.min_voiced_duration_per_syllable else 0.0
        return total_syllables, min_duration_budget, max_duration_budget

    def is_valid(self, context: ValidationContext) -> ValidationResult:
        params = context.validation_params
        if params.min_voiced_duration_per_syllable is None and params.max_voiced_duration_per_syllable is None:
            return ValidationResult(is_ok=True)

        total_voiced_samples = sum(s['end'] - s['start'] for s in context.filtered_speech_segments)
        actual_voiced_seconds = total_voiced_samples / context.sample_rate

        if params.use_word_level_duration_analysis:
            total_syllables, min_expected, max_expected = self._calculate_dynamic_duration_budgets(context)
            if total_syllables == 0:
                return ValidationResult(is_ok=True)

            if params.min_syllables_for_duration_validation is not None and total_syllables < params.min_syllables_for_duration_validation:
                logging.debug(f"Skipping duration validation for short text (syllables: {total_syllables} < threshold).")
                return ValidationResult(is_ok=True)

            if params.min_voiced_duration_per_syllable is not None and actual_voiced_seconds < min_expected:
                return ValidationResult(is_ok=False, reason=f"Voiced duration too short: {actual_voiced_seconds:.2f}s (min expected: {min_expected:.2f}s)")
            
            if params.max_voiced_duration_per_syllable is not None and actual_voiced_seconds > max_expected:
                return ValidationResult(is_ok=False, reason=f"Voiced duration too long: {actual_voiced_seconds:.2f}s (max budget: {max_expected:.2f}s)")

        else:
            syllable_count = context.syllable_count
            if syllable_count == 0:
                return ValidationResult(is_ok=True)
            
            if params.min_syllables_for_duration_validation is not None and syllable_count < params.min_syllables_for_duration_validation:
                logging.debug(f"Skipping duration validation for short text (syllables: {syllable_count} < threshold).")
                return ValidationResult(is_ok=True)
            
            ratio = actual_voiced_seconds / syllable_count

            if params.min_voiced_duration_per_syllable is not None and ratio < params.min_voiced_duration_per_syllable:
                return ValidationResult(is_ok=False, reason=f"Voiced duration too short: {ratio:.3f}s/syl (min: {params.min_voiced_duration_per_syllable:.3f}s/syl)")
            
            if params.max_voiced_duration_per_syllable is not None and ratio > params.max_voiced_duration_per_syllable:
                return ValidationResult(is_ok=False, reason=f"Voiced duration too long: {ratio:.3f}s/syl (max: {params.max_voiced_duration_per_syllable:.3f}s/syl)")

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
    Calculates the standard deviation of the spectral centroid across all voiced frames.
    Returns the float value or None if it cannot be computed.
    """
    if not context.filtered_speech_segments:
        return None

    try:
        # 1. Concatenate all voiced speech segments into a single tensor.
        voiced_audio = torch.cat(
            [context.original_waveform[..., seg['start']:seg['end']] for seg in context.filtered_speech_segments],
            dim=-1
        )

        # Use a smaller n_fft for better time resolution and to handle short segments.
        # 1024 is a good balance for speech at 22-24kHz. Hop length is typically n_fft // 4.
        n_fft = 1024
        hop_length = n_fft // 4

        if voiced_audio.numel() < n_fft:
            logging.warning("Not enough voiced audio to calculate spectral centroid.")
            return None

        # 2. Create the spectrogram for the entire voiced portion at once.
        spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=2 # Use power spectrogram for centroid calculation
        )
        spec = spectrogram_transform(voiced_audio)

        # If spec has a channel dimension, remove it for calculation.
        if spec.ndim == 3:
            spec = spec.squeeze(0) # Shape: [freq, time]

        # 3. Calculate frame-by-frame spectral centroid.
        # Ensure freqs tensor device matches spectrogram device.
        freqs = torch.linspace(0, context.sample_rate / 2, steps=spec.shape[0], device=spec.device)
        
        # Reshape for broadcasting: freqs -> [freq, 1], spec -> [freq, time]
        # Result of multiplication is [freq, time]
        numerator = torch.sum(freqs.unsqueeze(1) * spec, dim=0)
        denominator = torch.sum(spec, dim=0) + 1e-9

        # centroid_per_frame is now a 1D tensor of centroids, one for each time frame.
        centroid_per_frame = numerator / denominator

        # 4. Calculate the standard deviation of the frame-wise centroids.
        # We need at least 2 frames for a meaningful standard deviation.
        if centroid_per_frame.numel() < 2:
            return None

        std_dev = torch.std(centroid_per_frame).item()
        return std_dev

    except Exception as e:
        logging.warning(f"Spectral Centroid calculation failed with an internal error: {e}", exc_info=True)
        return None

class SpectralCentroidValidator(AbstractValidator):
    def is_valid(self, context: ValidationContext) -> ValidationResult:
        params = context.validation_params
        if params.min_spectral_centroid_std_dev is None:
            return ValidationResult(is_ok=True)

        if params.min_syllables_for_spectral_validation is not None:
            syllable_count = context.syllable_count
            if syllable_count < params.min_syllables_for_spectral_validation:
                logging.debug(
                    f"Skipping spectral centroid validation for short text "
                    f"(syllables: {syllable_count} < threshold: {params.min_syllables_for_spectral_validation})."
                )
                return ValidationResult(is_ok=True)

        std_dev = _calculate_spectral_centroid_std_dev(context)

        if std_dev is None:
            logging.debug("Spectral centroid could not be calculated. Skipping validation for this check.")
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