import torch
import logging
from chatterbox.tts import ChatterboxTTS

from tts_api.core.config import settings
from tts_api.core.models import ChatterboxParams
from tts_api.tts_engines.base import AbstractTTSEngine

def _noop_watermark(wav, sample_rate):
    """A replacement function that does nothing, effectively disabling the watermark."""
    logging.debug("Watermark disabled. Returning original audio.")
    return wav

class ChatterboxEngine(AbstractTTSEngine):
    def __init__(self):
        self._model: ChatterboxTTS | None = None

    def load_model(self):
        if self._model is None:
            logging.info("Chatterbox model not loaded. Initializing...")
            try:
                self._model = ChatterboxTTS.from_pretrained(device=settings.DEVICE)
                logging.info(f"Chatterbox model loaded on device: {settings.DEVICE}")
            except Exception as e:
                logging.critical(f"Failed to load Chatterbox model: {e}", exc_info=True)
                raise
    
    @property
    def sample_rate(self) -> int:
        if self._model is None:
            raise RuntimeError("Model is not loaded yet.")
        return self._model.sr

    def generate(self, text: str, params: ChatterboxParams, **kwargs) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("Cannot generate audio, model is not loaded.")
        
        # This endpoint now requires ref_audio, so the path is guaranteed to be in kwargs.
        ref_audio_path = kwargs.get("ref_audio_path")
        if not ref_audio_path:
            raise ValueError("Chatterbox engine requires a reference audio path for generation.")

        original_watermark_method = None
        if params.disable_watermark:
            logging.debug("Attempting to disable watermark via monkey-patch...")
            original_watermark_method = self._model.watermarker.apply_watermark
            self._model.watermarker.apply_watermark = _noop_watermark
        
        try:
            # The model's own .generate() method handles calling .prepare_conditionals()
            # when audio_prompt_path is provided. This is much cleaner.
            # The model's generate() also calls punc_norm() internally.
            wav_tensor = self._model.generate(
                text=text,
                audio_prompt_path=ref_audio_path,
                exaggeration=params.exaggeration,
                cfg_weight=params.cfg_weight,
                temperature=params.temperature,
            )
        finally:
            # CRITICAL: Always restore the original method to prevent side-effects.
            if original_watermark_method is not None:
                logging.debug("Restoring original watermark method.")
                self._model.watermarker.apply_watermark = original_watermark_method
            
        return wav_tensor

# Singleton instance
chatterbox_engine = ChatterboxEngine()