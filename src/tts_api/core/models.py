from enum import Enum
from pydantic import BaseModel, Field
from typing import Literal, List, Union, Optional

# --- Engine Specific Parameter Models ---

class ChatterboxParams(BaseModel):
    """Parameters specific to the Chatterbox TTS engine."""
    exaggeration: float = Field(0.5, ge=0.0, le=2.0, description="Emotion exaggeration.")
    temperature: float = Field(0.8, ge=0.01, le=5.0, description="Generation temperature.")
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0, description="CFG Weight.")
    disable_watermark: bool = Field(True, description="Disable Chatterbox's audio watermark.")
    use_analyzer: bool = Field(False, description="Enable the alignment analyzer for improved quality and robustness. May fail on some inputs.")

# --- Generic Request Component Models ---

class TextChunkingStrategy(str, Enum):
    PARAGRAPH = "paragraph"
    BALANCED = "balanced"
    GREEDY = "greedy"
    SIMPLE = "simple"

class TextProcessingOptions(BaseModel):
    """Options for cleaning and chunking the input text."""
    text_language: str = Field("en", description="Language code for sentence segmentation and normalization (e.g., 'en', 'de', 'es').")
    to_lowercase: bool = Field(False, description="Convert input text to lowercase.")
    remove_bracketed_text: bool = Field(False, description="Remove text in brackets [] and parentheses ().")
    use_nemo_normalizer: bool = Field(False, description="Use NeMo to normalize numbers, dates, currencies, etc., to words.")
    apply_advanced_cleaning: bool = Field(False, description="Apply advanced, heuristic cleaning rules like stutter removal ('b-but') and emphasis normalization ('*word*').")
    chunking_strategy: TextChunkingStrategy = Field(
        TextChunkingStrategy.PARAGRAPH, 
        description=(
            "'paragraph': Keeps paragraphs whole, falling back to 'balanced' for long ones (Recommended). "
            "'balanced': Optimally splits sentences for even chunk sizes. "
            "'greedy': Groups sentences greedily (faster, less even). "
            "'simple': Force-splits by character length."))
    shortness_penalty_factor: float = Field(2.0, ge=1.0, description="Factor to penalize short chunks in the 'balanced' strategy. Higher values avoid tiny chunks more aggressively.")
    ideal_chunk_length: int = Field(300, ge=50, le=1000, description="The target character length for chunks. Used by 'balanced' strategy.")
    max_chunk_length: int = Field(500, ge=50, le=1000, description="The absolute maximum character length for any text chunk.")

class ValidationParams(BaseModel):
    """Parameters for post-generation validation and retries."""
    max_silence_dbfs: Optional[float] = Field(
        -60.0, ge=-120.0, le=-10.0,  description="Fail if raw audio RMS is below this level (dBFS). Catches silent/dead outputs. Set to None to disable.")
    min_voiced_duration_per_syllable: Optional[float] = Field(
        0.150, ge=0.0, le=2.0,  description="Fail if final speech duration is less than (this * syllables) (seconds). Catches truncated audio. Set to None to disable.")
    max_voiced_duration_per_syllable: Optional[float] = Field(
        0.350, ge=0.0, le=5.0,  description="Fail if final speech duration is more than (this * syllables) (seconds). Catches rambling and hallucinated audio. Set to None to disable.")
    min_syllables_for_duration_validation: Optional[int] = Field(
        7, ge=1, description="Do not run the voiced duration-per-syllable validator if the text chunk has fewer syllables than this. Helps prevent false failures on very short text. Set to None to always run.")
    max_contiguous_silence_s: Optional[float] = Field(
        2.8, ge=0.0, le=10.0,  description="Fail if any silent gap between speech chunks is longer than this (seconds). Set to None to disable.")
    max_clipping_percentage: Optional[float] = Field(
        0.1, ge=0.0, le=100.0,  description="Fail if more than this percentage of speech samples are clipped. Catches distortion. Set to None to disable.")
    min_spectral_centroid_std_dev: Optional[float] = Field(
        250.0, ge=0.0, le=6000.0,  description="Fail if spectral centroid std dev is lower than this level (Hz). Catches monotonous, noisy audio. Set to None to disable.")

class AudioNormalizationMethod(str, Enum):
    EBU = "ebu"
    PEAK = "peak"

class ExportFormat(str, Enum):
    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"

class PostProcessingOptions(BaseModel):
    """Options for post-processing the generated audio."""
    denoise_with_vad: bool = Field(True, description="Master switch to enable VAD-based denoising.")
    vad_threshold: float = Field(0.5, ge=0.0, le=1.0, description="VAD confidence threshold for detecting speech.")
    vad_min_silence_ms: int = Field(100, ge=0, le=5000, description="Minimum silence duration (ms) required to split speech segments.")
    vad_min_speech_ms: int = Field(250, ge=0, le=1000, description="Minimum duration (ms) for a detected speech segment. Discards shorter segments as noise.")
    vad_speech_pad_ms: int = Field(50, ge=0, le=500, description="Padding (ms) added to the start/end of speech segments to prevent clipping.")
    vad_fade_ms: int = Field(10, ge=0, le=200, description="Duration (ms) of fade-in/out at speech boundaries for smooth transitions.")
    max_voiced_dynamic_range_db: Optional[float] = Field(20.0, ge=0.0, le=60.0, description="Segments with RMS energy this many dB below the loudest segment will be filtered out. Set to None to disable.")

    trim_segment_silence: bool = Field(
        True, description="Trim leading/trailing silence from each audio segment before concatenation. This is recommended when setting inter-segment silences to non-zero values.")
    inter_segment_silence_sentence_ms: int = Field(
        200, ge=0, le=5000, description="Silence duration (ms) to insert between segments that represent sentences within the same paragraph.")
    inter_segment_silence_paragraph_ms: int = Field(
        600, ge=0, le=5000, description="Silence duration (ms) to insert between segments that represent paragraph breaks.")
    inter_segment_silence_fallback_ms: int = Field(
        150, ge=0, le=5000, description="Silence duration (ms) to insert when a segment was split due to length, not a semantic boundary.")

    normalize_audio: bool = Field(True, description="Normalize audio loudness with ffmpeg.")
    normalize_method: AudioNormalizationMethod = Field(AudioNormalizationMethod.EBU, description="Normalization method.")
    normalize_level: float = Field(-18.0, ge=-70.0, le=-5.0, description="EBU Target Integrated Loudness (LUFS).")

    export_format: ExportFormat = Field(ExportFormat.MP3, description="Final audio format.")


# --- Main Request Models ---

class BaseTTSRequest(BaseModel):
    """Base model with parameters common to all TTS engines."""
    text: str = Field(..., description="The text to be converted to speech.", min_length=1)
    seed: Union[int, List[int]] = Field(0, description="Random seed or a list of seeds. If a list is provided, seeds are used sequentially for each generation candidate. A seed of 0 means use a random seed.")
    best_of: int = Field(1, ge=1, le=10, description="Generate multiple speech outputs and automatically return the one with the highest speech-to-audio-duration ratio. Higher values take longer but can improve quality.")
    max_retries: int = Field(1, ge=0, le=10, description="Number of times to retry a failed or low-quality generation for a single text chunk.")
    validation_params: ValidationParams = Field(default_factory=ValidationParams)
    text_processing: TextProcessingOptions = Field(default_factory=TextProcessingOptions)
    post_processing: PostProcessingOptions = Field(default_factory=PostProcessingOptions)

class ChatterboxTTSRequest(BaseTTSRequest):
    """The complete request model for the Chatterbox engine endpoint."""
    chatterbox_params: ChatterboxParams = Field(default_factory=ChatterboxParams)