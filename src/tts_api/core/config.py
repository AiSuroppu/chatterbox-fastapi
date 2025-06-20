import torch
from pydantic_settings import BaseSettings
from pathlib import Path

# Define the project's base directory to reliably locate the .cache folder
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_NEMO_CACHE_DIR = BASE_DIR / ".cache" / "nemo"

class Settings(BaseSettings):
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Generic model path dictionary for multiple engines
    # In .env file, you can set it like:
    # MODEL_PATHS='{"chatterbox": "/path/to/your/chatterbox_model"}'
    MODEL_PATHS: dict[str, str] = {}

    # List of engine names to load on startup.
    # In .env file: ENABLED_MODELS='["chatterbox"]'
    ENABLED_MODELS: list[str] = ["chatterbox"]

    # Path to a directory for NeMo's compiled grammar files.
    # Defaults to ./.cache/nemo within the project root.
    NEMO_CACHE_DIR: str = str(DEFAULT_NEMO_CACHE_DIR)

    # LRU cache size for the NeMo Normalizer instances.
    NEMO_NORMALIZER_CACHE_SIZE: int = 4

    # LRU cache size for the pysbd Segmenter instances.
    PYSBD_CACHE_SIZE: int = 4

    # LRU cache size for the pyphen.Pyphen dictionary instances.
    PYPHEN_CACHE_SIZE: int = 4

    # --- Chatterbox Specific Settings ---
    # The maximum number of text segments to process in a single batch for Chatterbox.
    CHATTERBOX_MAX_BATCH_SIZE: int = 2
    # PyTorch 2.0+ model compilation mode for Chatterbox (e.g., 'default', 'reduce-overhead', 'max-autotune').
    # Set to an empty string to disable compilation.
    CHATTERBOX_COMPILE_MODE: str = ""
    # The maximum number of voice embeddings to keep in the server-side cache.
    CHATTERBOX_VOICE_CACHE_SIZE: int = 10
    # Offload the S3 generation model to CPU after use to save VRAM.
    CHATTERBOX_OFFLOAD_S3GEN: bool = False
    # Offload the T3 (text-to-token) model to CPU after use to save VRAM.
    CHATTERBOX_OFFLOAD_T3: bool = False

    # --- FishSpeech Specific Settings ---
    # Path to the Text-to-Semantic (T2S) model checkpoint directory.
    FISHSPEECH_T2S_CHECKPOINT_PATH: str = "vendor/fish_speech/checkpoints/openaudio-s1-mini"
    # Path to the vocoder/decoder model checkpoint file.
    FISHSPEECH_DECODER_CHECKPOINT_PATH: str = "vendor/fish_speech/checkpoints/openaudio-s1-mini/codec.pth"
    # The configuration name for the vocoder/decoder model.
    FISHSPEECH_DECODER_CONFIG_NAME: str = "modded_dac_vq"
    # Enable PyTorch 2.0+ model compilation for the T2S model.
    FISHSPEECH_COMPILE: bool = False
    # Maximum number of cached reference voices (prompt tokens) to keep in memory.
    FISHSPEECH_VOICE_CACHE_SIZE: int = 10
    # Offload the Text-to-Semantic (LLaMA-based) model to the CPU when not in use to save VRAM.
    FISHSPEECH_OFFLOAD_T2S_MODEL: bool = False
    # Offload the Vocoder/Decoder (DAC) model to the CPU when not in use to save VRAM.
    FISHSPEECH_OFFLOAD_DECODER_MODEL: bool = False
    # Mini-batch size for the Text-to-Semantic (LLaMA) model.
    # NOTE: The current engine implementation does not support batching for this model yet.
    FISHSPEECH_T2S_BATCH_SIZE: int = 1
    # Mini-batch size for the vocoder/decoder model.
    FISHSPEECH_DECODER_BATCH_SIZE: int = 1

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()