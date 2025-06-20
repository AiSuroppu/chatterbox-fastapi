import logging
import hashlib
from typing import List, Optional, Tuple, Dict, Any
from cachetools import LRUCache
import torch
import torch.nn.functional as F

from fish_speech.models.text2semantic.inference import init_model as init_t2s_model, generate_long
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.inference_engine.reference_loader import ReferenceLoader

from tts_api.core.config import settings
from tts_api.core.models import FishSpeechParams
from tts_api.core.exceptions import InvalidVoiceTokenException
from tts_api.tts_engines.base import AbstractTTSEngine

class FishSpeechEngine(AbstractTTSEngine, ReferenceLoader):
    def __init__(self):
        AbstractTTSEngine.__init__(self)
        ReferenceLoader.__init__(self) # This will set up the torchaudio backend

        self._t2s_model = None
        self._t2s_decode_one_token = None
        self._decoder_model = None
        self._voice_cache: Optional[LRUCache] = None
        self.name = "fish_speech"

    @property
    def sample_rate(self) -> int:
        if self._decoder_model is None:
            raise RuntimeError("Decoder model is not loaded yet.")
        return self._decoder_model.sample_rate

    def load_model(self):
        if self._t2s_model is not None:
            return # Already loaded

        logging.info("FishSpeech models not loaded. Initializing...")
        try:
            precision = torch.half if settings.DEVICE == 'cuda' else torch.bfloat16
            
            # 1. Load Text-to-Semantic Model
            self._t2s_model, self._t2s_decode_one_token = init_t2s_model(
                checkpoint_path=settings.FISHSPEECH_T2S_CHECKPOINT_PATH,
                device=settings.DEVICE,
                precision=precision,
                compile=settings.FISHSPEECH_COMPILE
            )

            # 2. Load Vocoder/Decoder Model
            self._decoder_model = load_decoder_model(
                config_name=settings.FISHSPEECH_DECODER_CONFIG_NAME,
                checkpoint_path=settings.FISHSPEECH_DECODER_CHECKPOINT_PATH,
                device=settings.DEVICE
            )
            
            # 3. Initialize Voice Cache
            cache_size = settings.FISHSPEECH_VOICE_CACHE_SIZE
            if cache_size > 0:
                self._voice_cache = LRUCache(maxsize=cache_size)
                logging.info(f"FishSpeech voice cache enabled with size {cache_size}.")

            logging.info(f"FishSpeech models loaded successfully on device: {settings.DEVICE}")
        except Exception as e:
            logging.critical(f"Failed to load FishSpeech models: {e}", exc_info=True)
            raise

    def _get_token_from_audio(self, ref_audio_data: bytes) -> str:
        """Creates a deterministic token from audio content."""
        return hashlib.sha256(ref_audio_data).hexdigest()

    def _create_prompt_tokens_from_data(self, ref_audio_data: bytes) -> torch.Tensor:
        """Encodes reference audio bytes into semantic tokens (the 'voice embedding')."""
        # This logic comes from fish_speech/inference_engine/vq_manager.py
        audio_tensor = self.load_audio(ref_audio_data, self.sample_rate)
        audios = torch.from_numpy(audio_tensor).to(self._decoder_model.device)[None, None, :]
        audio_lengths = torch.tensor([audios.shape[2]], device=self._decoder_model.device, dtype=torch.long)
        
        prompt_tokens, _ = self._decoder_model.encode(audios, audio_lengths)
        return prompt_tokens

    def prepare_generation(
        self,
        params: FishSpeechParams,
        ref_audio_data: Optional[bytes]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Prepares the voice prompt tokens for generation, using the cache.
        The "voice embedding" for FishSpeech is the tensor of prompt_tokens.
        """
        if self._t2s_model is None or self._decoder_model is None:
            raise RuntimeError("Models are not loaded yet.")

        generation_kwargs = {}
        response_metadata = {}

        # Case 1: Client provides a token
        if self._voice_cache is not None and params.voice_cache_token:
            cached_prompt_tokens = self._voice_cache.get(params.voice_cache_token)
            if cached_prompt_tokens is not None:
                logging.info(f"Used cached voice with token: {params.voice_cache_token}")
                generation_kwargs["prompt_tokens"] = cached_prompt_tokens
                response_metadata["voice_cache_token"] = params.voice_cache_token
                return generation_kwargs, response_metadata
            else:
                raise InvalidVoiceTokenException(token=params.voice_cache_token)

        # Case 2: Client provides reference audio
        if ref_audio_data:
            prompt_tokens = self._create_prompt_tokens_from_data(ref_audio_data)
            
            if self._voice_cache is not None:
                token = self._get_token_from_audio(ref_audio_data)
                self._voice_cache[token] = prompt_tokens
                response_metadata["voice_cache_token"] = token
                logging.info(f"Created and cached new voice with token: {token}")

            generation_kwargs["prompt_tokens"] = prompt_tokens
            return generation_kwargs, response_metadata

        # Case 3: Neither provided. FishSpeech can run without a reference (zero-shot).
        logging.info("No reference audio or token provided. Using zero-shot generation.")
        generation_kwargs["prompt_tokens"] = None
        return generation_kwargs, response_metadata

    def generate(self, text_chunks: List[str], params: FishSpeechParams, **kwargs) -> List[Optional[torch.Tensor]]:
        if self._t2s_model is None:
            raise RuntimeError("Cannot generate, T2S model not loaded.")
        
        prompt_tokens = kwargs.get("prompt_tokens")
        
        # Phase 1: Generate semantic codes for all chunks (serially)
        all_semantic_codes = []
        for text in text_chunks:
            generator = generate_long(
                model=self._t2s_model,
                device=settings.DEVICE,
                decode_one_token=self._t2s_decode_one_token,
                text=text,
                prompt_text=None,
                prompt_tokens=[prompt_tokens] if prompt_tokens is not None else None,
                temperature=params.temperature,
                top_p=params.top_p,
                repetition_penalty=params.repetition_penalty,
                max_new_tokens=params.max_new_tokens,
            )
            
            codes_for_chunk = None
            for response in generator:
                if response.action == "sample":
                    codes_for_chunk = response.codes
                elif response.action == "next":
                    break
            
            if codes_for_chunk is None:
                logging.error(f"Failed to generate semantic codes for text: '{text[:50]}...'")
            all_semantic_codes.append(codes_for_chunk)

        # Phase 2: Batch-decode all valid semantic codes
        valid_codes = [code for code in all_semantic_codes if code is not None]
        if not valid_codes:
            logging.warning("No valid semantic codes were generated for any chunk.")
            return [None] * len(text_chunks)

        # Pad all valid codes to the same length for batching
        max_len = max(c.shape[1] for c in valid_codes)
        code_lengths = torch.tensor([c.shape[1] for c in valid_codes], device=settings.DEVICE)
        
        padded_codes = []
        for code in valid_codes:
            pad_len = max_len - code.shape[1]
            padded_code = F.pad(code, (0, pad_len), "constant", 0) # Pad the sequence dimension
            padded_codes.append(padded_code)
        
        batched_codes = torch.stack(padded_codes, dim=0).to(settings.DEVICE)

        logging.info(f"Batch-decoding {len(valid_codes)} semantic code sequences with the vocoder.")
        batched_waveforms, _ = self._decoder_model.decode(batched_codes, feature_lengths=code_lengths)
        
        # Phase 3: Reconstruct the final list, placing waveforms in their original positions
        final_waveforms = [None] * len(text_chunks)
        valid_idx_counter = 0
        for i, code in enumerate(all_semantic_codes):
            if code is not None:
                # The i-th original chunk was valid, so take the next available waveform
                final_waveforms[i] = batched_waveforms[valid_idx_counter].cpu()
                valid_idx_counter += 1

        return final_waveforms


# Singleton instance for the API to use
fish_speech_engine = FishSpeechEngine()