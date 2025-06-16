import torch
import logging
import hashlib
import tempfile
import os
from typing import List, Optional, Tuple, Dict, Any
from cachetools import LRUCache
from chatterbox.tts import ChatterboxTTS

from tts_api.core.config import settings
from tts_api.core.models import ChatterboxParams
from tts_api.core.exceptions import InvalidVoiceTokenException
from tts_api.tts_engines.base import AbstractTTSEngine

def _noop_watermark(wav, sample_rate):
    """A replacement function that does nothing, effectively disabling the watermark."""
    logging.debug("Watermark disabled. Returning original audio.")
    return wav

class ChatterboxEngine(AbstractTTSEngine):
    def __init__(self):
        self._model: Optional[ChatterboxTTS] = None
        self._voice_cache: Optional[LRUCache] = None
        self.name = "chatterbox"

    def load_model(self):
        if self._model is None:
            logging.info("Chatterbox model not loaded. Initializing...")
            try:
                cache_size = settings.CHATTERBOX_VOICE_CACHE_SIZE
                if cache_size > 0:
                    self._voice_cache = LRUCache(maxsize=cache_size)
                    logging.info(f"Chatterbox voice cache enabled with size {cache_size}.")
                
                compile_mode = settings.CHATTERBOX_COMPILE_MODE.strip() or None
                self._model = ChatterboxTTS.from_pretrained(device=settings.DEVICE, compile_mode=compile_mode)
                logging.info(f"Chatterbox model loaded on device: {settings.DEVICE}")
            except Exception as e:
                logging.critical(f"Failed to load Chatterbox model: {e}", exc_info=True)
                raise

    def _create_embedding_from_data(self, ref_audio_data: bytes, exaggeration: float):
        """Helper to create a voice embedding from raw audio bytes."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_file.write(ref_audio_data)
            tmp_filepath = tmp_file.name
        try:
            embedding = self._model.create_voice_embedding(
                wav_fpath=tmp_filepath,
                exaggeration=exaggeration
            )
            return embedding
        finally:
            os.remove(tmp_filepath)
    
    def _get_token_from_audio(self, ref_audio_data: bytes, exaggeration: float) -> str:
        """Creates a deterministic token from audio content and a parameter."""
        audio_hash = hashlib.sha256(ref_audio_data).hexdigest()
        return f"{audio_hash}-{exaggeration}"

    def prepare_generation(
        self,
        params: ChatterboxParams,
        ref_audio_data: Optional[bytes]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Prepares the voice embedding for generation, utilizing a cache if enabled.
        This method ensures `generate` always receives a pre-computed embedding.
        """
        if self._model is None:
            raise RuntimeError("Model is not loaded yet.")

        generation_kwargs = {}
        response_metadata = {}

        # Case 1: Client provides a token. This is the primary path for cached voices.
        if self._voice_cache is not None and params.voice_cache_token:
            cached_embedding = self._voice_cache.get(params.voice_cache_token)
            
            if cached_embedding:
                logging.info(f"Used cached voice with token: {params.voice_cache_token}")
                generation_kwargs["voice_embedding_cache"] = cached_embedding
                # Always return the token if it was successfully used to confirm its validity.
                response_metadata["voice_cache_token"] = params.voice_cache_token
                return generation_kwargs, response_metadata
            else:
                # The token was provided, but it's invalid/expired. This is a specific failure.
                # The client made a specific claim (this token is valid) which is false.
                raise InvalidVoiceTokenException(token=params.voice_cache_token)

        # Case 2: Client provides reference audio. This is the primary path for new voices.
        if ref_audio_data:
            # If cache is enabled, create embedding, store it, and return a new token.
            if self._voice_cache is not None:
                # This engine's token is based on the audio content.
                token = self._get_token_from_audio(ref_audio_data, params.exaggeration)

                if token in self._voice_cache:
                    embedding = self._voice_cache.get(token)
                    logging.info(f"Found matching voice in cache. Reusing token: {token}")
                else:
                    logging.info(f"Creating and caching new voice embedding with token: {token}")
                    embedding = self._create_embedding_from_data(ref_audio_data, params.exaggeration)
                    self._voice_cache[token] = embedding
                
                response_metadata["voice_cache_token"] = token
                generation_kwargs["voice_embedding_cache"] = embedding
            else:
                # Caching is disabled: create a one-time embedding
                logging.info("Voice cache disabled. Creating a temporary voice embedding.")
                embedding = self._create_embedding_from_data(ref_audio_data, params.exaggeration)
                generation_kwargs["voice_embedding_cache"] = embedding
            
            return generation_kwargs, response_metadata

        # Case 3: Client provides neither a token nor reference audio. This is a clear client error.
        raise ValueError("Chatterbox is a voice cloning engine and requires either a `ref_audio` file or a valid `voice_cache_token`.")

    @property
    def sample_rate(self) -> int:
        if self._model is None:
            raise RuntimeError("Model is not loaded yet.")
        return self._model.sr

    def generate(self, text_chunks: List[str], params: ChatterboxParams, **kwargs) -> List[torch.Tensor]:
        if self._model is None:
            raise RuntimeError("Cannot generate audio, model is not loaded.")
        
        voice_embedding_cache = kwargs.get("voice_embedding_cache")
        if not voice_embedding_cache:
            # This should not happen if prepare_generation is called correctly.
            raise ValueError("`generate` was called without a `voice_embedding_cache`. This indicates a service-layer error.")

        original_watermark_method = None
        try:
            if params.disable_watermark:
                original_watermark_method = self._model.watermarker.apply_watermark
                self._model.watermarker.apply_watermark = _noop_watermark
            
            logging.debug(f"  Engine processing {len(text_chunks)} text chunks with a batch of size {settings.CHATTERBOX_MAX_BATCH_SIZE}")
            all_waveforms = self._model.generate(
                text=text_chunks,
                audio_prompt_path=None, # Explicitly set to None
                exaggeration=params.exaggeration,
                cfg_weight=params.cfg_weight,
                temperature=params.temperature,
                use_analyzer=params.use_analyzer,
                voice_embedding_cache=voice_embedding_cache,
                offload_s3gen=settings.CHATTERBOX_OFFLOAD_S3GEN,
                offload_t3=settings.CHATTERBOX_OFFLOAD_T3,
                batch_size=settings.CHATTERBOX_MAX_BATCH_SIZE
            )
            if not isinstance(all_waveforms, list):
                all_waveforms = [all_waveforms]  # Ensure we always have a list
            
            return all_waveforms

        finally:
            if original_watermark_method:
                self._model.watermarker.apply_watermark = original_watermark_method

# Singleton instance
chatterbox_engine = ChatterboxEngine()