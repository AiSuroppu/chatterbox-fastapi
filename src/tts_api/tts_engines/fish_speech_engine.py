import logging
import hashlib
from typing import List, Optional, Tuple, Dict, Any
from cachetools import LRUCache
from contextlib import contextmanager
import torch
import torch.nn.functional as F

from fish_speech.models.text2semantic.inference import init_model as init_t2s_model, generate_long, generate_t2s_batch
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.inference_engine.reference_loader import ReferenceLoader

from tts_api.core.config import settings
from tts_api.core.models import FishSpeechParams
from tts_api.core.exceptions import InvalidVoiceTokenException, ClientRequestError, EngineExecutionError, EngineLoadError
from tts_api.tts_engines.base import AbstractTTSEngine

@contextmanager
def model_context(model: torch.nn.Module, target_device: str, offload_enabled: bool):
    """A context manager to move a model to a target device and back."""
    original_device = next(model.parameters()).device
    
    if not offload_enabled or str(original_device) == target_device:
        # If offloading is disabled or model is already on the target device, do nothing.
        yield
        return

    try:
        logging.debug(f"Moving model {type(model).__name__} to {target_device} for inference.")
        model.to(target_device)
        # On CUDA, an empty cache can free up memory from the other model
        if 'cuda' in target_device:
            torch.cuda.empty_cache()
        yield
    finally:
        logging.debug(f"Offloading model {type(model).__name__} back to {original_device}.")
        model.to(original_device)

class FishSpeechEngine(AbstractTTSEngine, ReferenceLoader):
    def __init__(self):
        AbstractTTSEngine.__init__(self)
        ReferenceLoader.__init__(self)

        self._t2s_model = None
        self._t2s_decode_one_token = None
        self._decoder_model = None
        self._voice_cache: Optional[LRUCache] = None
        self.name = "fish_speech"
        self._t2s_offload_device = "cpu" if settings.FISHSPEECH_OFFLOAD_T2S_MODEL else settings.DEVICE
        self._decoder_offload_device = "cpu" if settings.FISHSPEECH_OFFLOAD_DECODER_MODEL else settings.DEVICE

    @property
    def sample_rate(self) -> int:
        if self._decoder_model is None:
            raise RuntimeError("Decoder model is not loaded yet.")
        return self._decoder_model.sample_rate

    def load_model(self):
        if self._t2s_model is not None:
            return

        logging.info("FishSpeech models not loaded. Initializing...")
        try:
            # Use bfloat16 for Ampere+ GPUs, half for others
            precision = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.half
            
            # 1. Load Text-to-Semantic Model
            self._t2s_model, self._t2s_decode_one_token = init_t2s_model(
                checkpoint_path=settings.FISHSPEECH_T2S_CHECKPOINT_PATH,
                device=settings.DEVICE,
                precision=precision,
                compile=settings.FISHSPEECH_COMPILE
            )
            # Offload if configured
            if settings.FISHSPEECH_OFFLOAD_T2S_MODEL:
                logging.info("Offloading T2S model to CPU.")
                self._t2s_model.to(self._t2s_offload_device)

            # 2. Load Vocoder/Decoder Model
            self._decoder_model = load_decoder_model(
                config_name=settings.FISHSPEECH_DECODER_CONFIG_NAME,
                checkpoint_path=settings.FISHSPEECH_DECODER_CHECKPOINT_PATH,
                device=settings.DEVICE
            )
            # Offload if configured
            if settings.FISHSPEECH_OFFLOAD_DECODER_MODEL:
                logging.info("Offloading Decoder model to CPU.")
                self._decoder_model.to(self._decoder_offload_device)
            
            # 3. Initialize Voice Cache
            cache_size = settings.FISHSPEECH_VOICE_CACHE_SIZE
            if cache_size > 0:
                self._voice_cache = LRUCache(maxsize=cache_size)
                logging.info(f"FishSpeech voice cache enabled with size {cache_size}.")

            logging.info(f"FishSpeech models loaded successfully on device: {settings.DEVICE}")
        except Exception as e:
            raise EngineLoadError(f"Failed to load Fish-Speech models: {e}") from e

    def _create_prompt_tokens_from_data(self, ref_audio_data: bytes) -> torch.Tensor:
        """Encodes reference audio bytes into semantic tokens (the 'voice embedding')."""
        with torch.inference_mode(), \
             model_context(self._t2s_model, self._t2s_offload_device, settings.FISHSPEECH_OFFLOAD_T2S_MODEL), \
             model_context(self._decoder_model, settings.DEVICE, settings.FISHSPEECH_OFFLOAD_DECODER_MODEL):

            audio_tensor = self.load_audio(ref_audio_data, self.sample_rate)
            
            # Ensure tensors are created on the same device as the model
            model_device = next(self._decoder_model.parameters()).device
            audios = torch.from_numpy(audio_tensor).to(model_device)[None, None, :]
            audio_lengths = torch.tensor([audios.shape[2]], device=model_device, dtype=torch.long)
            
            prompt_tokens, _ = self._decoder_model.encode(audios, audio_lengths)
            if prompt_tokens.ndim == 3:
                prompt_tokens = prompt_tokens[0]
        return prompt_tokens

    def prepare_generation(
        self,
        params: FishSpeechParams,
        ref_audio_data: Optional[bytes],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Prepares the voice prompt tokens and text for generation, using the cache.
        The "voice embedding" for FishSpeech is the tensor of prompt_tokens and its transcription.
        """
        if self._t2s_model is None or self._decoder_model is None:
            raise RuntimeError("Models are not loaded yet.")

        generation_kwargs = {}
        response_metadata = {}

        # Case 1: Client provides a token
        if self._voice_cache is not None and params.voice_cache_token:
            # The cache now stores a tuple (prompt_tokens, prompt_text)
            cached_data = self._voice_cache.get(params.voice_cache_token)
            if cached_data is not None:
                prompt_tokens, prompt_text = cached_data
                logging.info(f"Used cached voice with token: {params.voice_cache_token}")
                generation_kwargs["prompt_tokens"] = prompt_tokens
                generation_kwargs["prompt_text"] = prompt_text
                response_metadata["voice_cache_token"] = params.voice_cache_token
                return generation_kwargs, response_metadata
            else:
                raise InvalidVoiceTokenException(token=params.voice_cache_token)

        # Case 2: Client provides reference audio and text
        if ref_audio_data:
            if not params.ref_text:
                raise ClientRequestError("Reference audio was provided, but the required 'ref_text' (transcription) is missing in the request.")

            prompt_tokens = self._create_prompt_tokens_from_data(ref_audio_data)
            
            if self._voice_cache is not None:
                # The token must be a composite of audio and text to be unique
                token_hash = hashlib.sha256()
                token_hash.update(ref_audio_data)
                token_hash.update(params.ref_text.encode('utf-8'))
                token = token_hash.hexdigest()

                # Cache the tuple of (tokens, text)
                self._voice_cache[token] = (prompt_tokens, params.ref_text)
                response_metadata["voice_cache_token"] = token
                logging.info(f"Created and cached new voice with token: {token}")

            generation_kwargs["prompt_tokens"] = prompt_tokens
            generation_kwargs["prompt_text"] = params.ref_text
            return generation_kwargs, response_metadata

        # Case 3: Neither provided. FishSpeech can run without a reference (zero-shot).
        logging.info("No reference audio or token provided. Using zero-shot generation.")
        generation_kwargs["prompt_tokens"] = None
        generation_kwargs["prompt_text"] = None
        return generation_kwargs, response_metadata

    def generate(self, text_chunks: List[str], params: FishSpeechParams, **kwargs) -> List[Optional[torch.Tensor]]:
        if self._t2s_model is None or self._decoder_model is None:
            raise EngineExecutionError("Cannot generate, Fish-Speech models not loaded.")

        try:
            prompt_tokens = kwargs.get("prompt_tokens")
            prompt_text = kwargs.get("prompt_text")

            if prompt_tokens is not None:
                prompt_tokens = prompt_tokens.to(settings.DEVICE)
                
            # Sort text chunks by length to optimize T2S batching by minimizing padding.
            # We store original indices to restore the order after T2S generation.
            indexed_text_chunks = sorted(
                enumerate(text_chunks),
                key=lambda item: len(item[1]),
                reverse=True
            )
            sorted_texts = [item[1] for item in indexed_text_chunks]
            original_indices_map = [item[0] for item in indexed_text_chunks]

            # Phase 1: Generate semantic codes with T2S model (LLaMA)
            t2s_batch_size = settings.FISHSPEECH_T2S_BATCH_SIZE

            sorted_semantic_codes = []
            with torch.inference_mode(), \
                model_context(self._decoder_model, self._decoder_offload_device, settings.FISHSPEECH_OFFLOAD_DECODER_MODEL), \
                model_context(self._t2s_model, settings.DEVICE, settings.FISHSPEECH_OFFLOAD_T2S_MODEL):

                logging.info(f"Generating semantic codes for {len(sorted_texts)} chunks with T2S model...")
                for i in range(0, len(sorted_texts), t2s_batch_size):
                    batch_texts = sorted_texts[i:i + t2s_batch_size]
                    logging.info(f"Processing T2S batch of size {len(batch_texts)}")

                    # Call the batched generation function
                    codes_for_batch = generate_t2s_batch(
                        model=self._t2s_model,
                        device=settings.DEVICE,
                        texts=batch_texts,
                        prompt_text=prompt_text,
                        prompt_tokens=prompt_tokens,
                        temperature=params.temperature,
                        top_p=params.top_p,
                        repetition_penalty=params.repetition_penalty,
                        max_new_tokens=params.max_new_tokens,
                    )
                    
                    sorted_semantic_codes.extend(codes_for_batch)

            # Restore the original order of the generated codes before decoding
            all_semantic_codes = [None] * len(text_chunks)
            for sorted_idx, original_idx in enumerate(original_indices_map):
                if sorted_idx < len(sorted_semantic_codes):
                    all_semantic_codes[original_idx] = sorted_semantic_codes[sorted_idx]

            # Phase 2: Batch-decode all valid semantic codes with Vocoder
            valid_codes_with_indices = [
                (i, code) for i, code in enumerate(all_semantic_codes) if code is not None
            ]
            if not valid_codes_with_indices:
                logging.warning("No valid semantic codes were generated for any chunk.")
                return [None] * len(text_chunks)
            
            # Sort by length in descending order for efficient vocoder batching
            valid_codes_with_indices.sort(key=lambda item: item[1].shape[1], reverse=True)
            
            final_waveforms = [None] * len(text_chunks)
            decoder_batch_size = settings.FISHSPEECH_DECODER_BATCH_SIZE

            with torch.inference_mode(), \
                model_context(self._t2s_model, self._t2s_offload_device, settings.FISHSPEECH_OFFLOAD_T2S_MODEL), \
                model_context(self._decoder_model, settings.DEVICE, settings.FISHSPEECH_OFFLOAD_DECODER_MODEL):

                logging.info(f"Decoding {len(valid_codes_with_indices)} semantic code sequences in batches of up to {decoder_batch_size}.")
                
                for i in range(0, len(valid_codes_with_indices), decoder_batch_size):
                    batch_of_indexed_codes = valid_codes_with_indices[i:i + decoder_batch_size]
                    original_indices = [item[0] for item in batch_of_indexed_codes]
                    codes_to_decode = [item[1] for item in batch_of_indexed_codes]
                    
                    max_len = codes_to_decode[0].shape[1]
                    
                    padded_codes = []
                    for code in codes_to_decode:
                        pad_len = max_len - code.shape[1]
                        padded_code = F.pad(code, (0, pad_len), "constant", 0)
                        padded_codes.append(padded_code)
                    
                    batched_codes = torch.stack(padded_codes, dim=0).to(settings.DEVICE)
                    code_lengths = torch.tensor([c.shape[1] for c in codes_to_decode], device=settings.DEVICE)

                    decoded_waveforms_batch, _ = self._decoder_model.decode(batched_codes, feature_lengths=code_lengths)
                    decoded_waveforms_batch = decoded_waveforms_batch.cpu()

                    # Place the decoded waveforms back into the correct positions in the final list
                    for original_idx, waveform in zip(original_indices, decoded_waveforms_batch):
                        final_waveforms[original_idx] = waveform

            return final_waveforms
        except Exception as e:
            # Wrap any unexpected internal error.
            raise EngineExecutionError(f"Fish-Speech model failed during generation: {e}") from e


# Singleton instance for the API to use
fish_speech_engine = FishSpeechEngine()