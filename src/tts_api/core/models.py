from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Union, Optional


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
    normalize_scene_breaks: bool = Field(
        True, description="Collapse character sequences that appear to be scene breaks into paragraph breaks."
    )
    min_scene_break_length: int = Field(
        3, ge=2, description=(
            "A line must contain at least this many non-whitespace characters to be considered a scene break. "
            "Helps prevent stray punctuation or short emphasis lines from being misinterpreted."
        )
    )
    normalize_repeated_punctuation: bool = Field(
        True, description=(
            "Enables a multi-stage process to normalize repeated and interleaved punctuation "
            "into TTS-friendly forms (e.g., '?!?!' -> '?!', '---' -> '—', '....' -> '...')."
        )
    )
    normalize_emphasis: bool = Field(
        True, description="Remove emphasis markers like '*' and '_' from around words (e.g., '*word*' -> 'word')."
    )
    normalize_stuttering: bool = Field(
        True, description="Collapse stuttering patterns (e.g., 'w-w-what' -> 'what')."
    )
    max_repeated_alpha_chars: Optional[int] = Field(
        4, ge=1, description="Truncate sequences of a repeated letter to this length (e.g., 'Ahhhhhh' -> 'Ahhh' with a value of 3). Set to None to disable."
    )
    max_repeated_symbol_chars: Optional[int] = Field(
        1, ge=1, description=(
            "Truncate sequences of any other repeated symbol (not covered by punctuation rules, e.g., '###', '***', '$$$') to this length. "
            "This is a general fallback rule that runs after specific punctuation normalization. Set to None to disable."
        )
    )
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

class ValidationOptions(BaseModel):
    """Parameters for post-generation validation and retries."""
    max_silence_dbfs: Optional[float] = Field(
        -60.0, ge=-120.0, le=-10.0,  description="Fail if raw audio RMS is below this level (dBFS). Catches silent/dead outputs. Set to None to disable.")
    max_contiguous_silence_s: Optional[float] = Field(
        2.8, ge=0.0, le=10.0,  description="Fail if any silent gap between speech chunks is longer than this (seconds). Catches missing speech, VAD failures and unnatural silences. Set to None to disable.")
    
    # Expected speech duration validation
    min_voiced_duration_per_syllable: Optional[float] = Field(
        0.150, ge=0.0, le=2.0,  description="Fail if final speech duration is less than (this * syllables) (seconds). Catches truncated audio. Set to None to disable.")
    max_voiced_duration_per_syllable: Optional[float] = Field(
        0.350, ge=0.0, le=5.0,  description=(
            "Fail if final speech duration is more than (this * syllables) (seconds). Catches rambling and hallucinated audio. Set to None to disable."
            "This value is used for normal-complexity words when 'use_word_level_duration_analysis' is enabled, otherwise, it is used as the global maximum."))
    min_syllables_for_duration_validation: Optional[int] = Field(
        7, ge=1, description="Do not run the voiced duration-per-syllable validator if the text chunk has fewer syllables than this. Helps prevent false failures on very short text. Set to None to always run.")
    # -- Word complexity analysis
    use_word_level_duration_analysis: bool = Field(
        True, description=(
            "Enables a word-by-word analysis to calculate the maximum speech duration per syllable budget. "
            "This allows for long onomatopoeia (e.g., 'Ahhhhhhhh') in the same sentence as normal text without causing a validation failure. If false, global 'max_voiced_duration_per_syllable' is used."))
    min_word_len_for_low_complexity_analysis: int = Field(
        7, ge=3, le=20, description=(
            "A word must have at least this many characters to be eligible for low-complexity analysis. "
            "This is intentionally set to a higher value by default to prioritize avoiding false positives on common, short English words."))
    low_complexity_log_variety_threshold: float = Field(
        1.6, ge=0.0, le=5.0, description=(
            "The Log-Normalized Variety (LNV) threshold used in word-level analysis. Words with an LNV below this value are considered low-complexity. LNV = unique_chars / log(word_length). "
            "Since analysis is restricted to longer words by 'min_word_len_for_low_complexity_analysis', this threshold is safer from misclassifying common words."))
    low_complexity_chars_per_syllable: float = Field(
        4.0, ge=1.0, le=10.0, description=(
            "When a word is identified as low-complexity (e.g., 'Ahhhhh'), its syllable count for the max duration budget is estimated as (word_length / this_value). This provides a scalable budget for onomatopoeia and other expressive sounds."))
    low_complexity_max_duration_per_syllable: float = Field(
        0.600, ge=0.0, le=10.0, description="The relaxed maximum speech duration per syllable for words identified as low-complexity (e.g., onomatopoeia) when word-level analysis is enabled.")
    
    # Audio quality validation
    max_clipping_percentage: Optional[float] = Field(
        0.1, ge=0.0, le=100.0,  description="Fail if more than this percentage of speech samples are clipped. Catches distortion. Set to None to disable.")
    min_spectral_centroid_std_dev: Optional[float] = Field(
        1800.0, ge=0.0, le=6000.0,  description="Fail if spectral centroid std dev is lower than this level (Hz). Catches monotonous, noisy audio. Set to None to disable.")
    min_syllables_for_spectral_validation: Optional[int] = Field(
        7, ge=1, description="Do not run the spectral centroid validator if the text chunk has fewer syllables than this. Helps prevent false failures on very short text (e.g., 'Yes', 'Okay'). Set to None to always run.")

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

    lead_in_silence_ms: Optional[int] = Field(
        None, ge=0, le=10000, description=(
            "Desired duration of silence (in milliseconds) at the beginning of the final audio. "
            "If set, this will override any existing silence from the model or VAD. "
            "If left as null (the default), the original lead-in silence is preserved (after VAD if enabled). "
            "A value of 0 will trim all lead-in silence."))
    lead_out_silence_ms: Optional[int] = Field(
        None, ge=0, le=10000, description=(
            "Desired duration of silence (in milliseconds) at the end of the final audio. "
            "If set, this will override any existing silence from the model or VAD. "
            "If left as null (the default), the original lead-out silence is preserved (after VAD if enabled). "
            "A value of 0 will trim all lead-out silence."))

    export_format: ExportFormat = Field(ExportFormat.MP3, description="Final audio format.")


# --- Main Request Models ---

class BaseTTSRequest(BaseModel):
    """Base model with parameters common to all TTS engines."""
    text: str = Field(..., description="The text to be converted to speech.", min_length=1)
    seed: Union[int, List[int]] = Field(0, description="Random seed or a list of seeds. If a list is provided, seeds are used sequentially for each generation candidate. A seed of 0 means use a random seed.")
    best_of: int = Field(1, ge=1, le=10, description="Generate multiple speech outputs and automatically return the one with the highest speech-to-audio-duration ratio. Higher values take longer but can improve quality.")
    max_retries: int = Field(1, ge=0, le=20, description="Number of times to retry a failed or low-quality generation for a single text chunk.")
    validation: ValidationOptions = ValidationOptions()
    text_processing: TextProcessingOptions = TextProcessingOptions()
    post_processing: PostProcessingOptions = PostProcessingOptions()


# --- Engine Specific Parameter Models ---

class ChatterboxParams(BaseModel):
    """Parameters specific to the Chatterbox TTS engine."""
    exaggeration: float = Field(0.5, ge=0.0, le=2.0, description="Emotion exaggeration.")
    temperature: float = Field(0.8, ge=0.01, le=5.0, description="Generation temperature.")
    cfg_weight: float = Field(0.5, ge=0.0, le=1.0, description="CFG Weight.")
    disable_watermark: bool = Field(True, description="Disable Chatterbox's audio watermark.")
    use_analyzer: bool = Field(False, description="Enable the alignment analyzer for improved quality and robustness. May fail on some inputs.")
    voice_cache_token: Optional[str] = Field(
        None, description="A token for a server-side cached voice. If provided, the 'ref_audio' file is ignored. The server returns this token in the 'X-Generation-Metadata' header after a successful generation with a new voice.")

class ChatterboxTTSRequest(BaseTTSRequest):
    """The complete request model for the Chatterbox engine endpoint."""
    chatterbox_params: ChatterboxParams = ChatterboxParams()


class FishSpeechParams(BaseModel):
    """Parameters specific to the FishSpeech TTS engine."""
    temperature: float = Field(0.8, ge=0.01, le=2.0, description="Generation temperature. Higher values are more random.")
    top_p: float = Field(0.8, ge=0.0, le=1.0, description="Top-P sampling threshold.")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Penalty for repeating tokens.")
    max_new_tokens: int = Field(1024, ge=1, le=4096, description="Maximum number of semantic tokens to generate per chunk.")
    ref_text: Optional[str] = Field(None, description="The transcription of the reference audio. Required for voice cloning.")
    voice_cache_token: Optional[str] = Field(None, description="A token for a server-side cached voice (encoded reference audio).")

class FishSpeechTTSRequest(BaseTTSRequest):
    """The complete request model for the FishSpeech engine endpoint."""
    fish_speech_params: FishSpeechParams = FishSpeechParams()