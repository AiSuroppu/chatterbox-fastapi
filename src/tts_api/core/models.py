from enum import Enum
from pydantic import BaseModel, Field
from typing import Literal, List, Union

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
            "'simple': Force-splits by character length."
        )
    )
    shortness_penalty_factor: float = Field(2.0, ge=1.0, description="Factor to penalize short chunks in the 'balanced' strategy. Higher values avoid tiny chunks more aggressively.")
    ideal_chunk_length: int = Field(300, ge=50, le=1000, description="The target character length for chunks. Used by 'balanced' strategy.")
    max_chunk_length: int = Field(500, ge=50, le=1000, description="The absolute maximum character length for any text chunk.")

class ValidationParams(BaseModel):
    """Parameters for post-generation validation and retries."""
    pass


class AudioNormalizationMethod(str, Enum):
    EBU = "ebu"
    PEAK = "peak"

class ExportFormat(str, Enum):
    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"

class PostProcessingOptions(BaseModel):
    """Options for post-processing the generated audio."""
    denoise_with_vad: bool = Field(True, description="Denoise silent segments using Silero-VAD.")
    vad_threshold: float = Field(0.5, ge=0.0, le=1.0, description="VAD confidence threshold for detecting speech.")
    vad_min_silence_ms: int = Field(100, ge=0, le=2000, description="Minimum silence duration (ms) to be replaced.")
    vad_speech_pad_ms: int = Field(50, ge=0, le=500, description="Padding (ms) added to the start/end of speech.")
    vad_fade_ms: int = Field(10, ge=0, le=200, description="Duration (ms) of fade-in/out at speech boundaries for smooth transitions.")
    normalize_audio: bool = Field(True, description="Normalize audio loudness with ffmpeg.")
    normalize_method: AudioNormalizationMethod = Field(AudioNormalizationMethod.EBU, description="Normalization method.")
    normalize_level: float = Field(-18.0, ge=-70.0, le=-5.0, description="EBU Target Integrated Loudness (I).")
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