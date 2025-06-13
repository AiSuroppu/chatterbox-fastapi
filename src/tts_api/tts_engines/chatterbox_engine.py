import torch
import logging
from typing import List
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
        self.name = "chatterbox"

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

    def generate(self, text_chunks: List[str], params: ChatterboxParams, **kwargs) -> List[torch.Tensor]:
        if self._model is None:
            raise RuntimeError("Cannot generate audio, model is not loaded.")
        
        ref_audio_path = kwargs.get("ref_audio_path")
        if not ref_audio_path:
            raise ValueError("Chatterbox engine requires a reference audio path for generation.")

        original_watermark_method = None
        if params.disable_watermark:
            logging.debug("Attempting to disable watermark via monkey-patch...")
            original_watermark_method = self._model.watermarker.apply_watermark
            self._model.watermarker.apply_watermark = _noop_watermark
        
        try:
            all_waveforms = []
            max_batch_size = settings.CHATTERBOX_MAX_BATCH_SIZE
            num_chunks = len(text_chunks)

            for i in range(0, num_chunks, max_batch_size):
                batch = text_chunks[i:i + max_batch_size]
                logging.debug(f"  Engine processing batch of size {len(batch)}")
                
                # The underlying library's generate method handles the batch
                waveform_batch = self._model.generate(
                    text=batch,
                    audio_prompt_path=ref_audio_path,
                    exaggeration=params.exaggeration,
                    cfg_weight=params.cfg_weight,
                    temperature=params.temperature,
                )
                if not isinstance(waveform_batch, list):
                    waveform_batch = [waveform_batch]  # Ensure we always have a list
                all_waveforms.extend(waveform_batch)
            
            return all_waveforms

        finally:
            # CRITICAL: Always restore the original method to prevent side-effects.
            if original_watermark_method is not None:
                logging.debug("Restoring original watermark method.")
                self._model.watermarker.apply_watermark = original_watermark_method

# Singleton instance
chatterbox_engine = ChatterboxEngine()