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
    target_device_obj = torch.device(target_device)

    # Do nothing if offloading is disabled or the model is already on the target device
    if not offload_enabled or original_device == target_device_obj:
        yield
        return

    try:
        logging.debug(f"Moving model {type(model).__name__} to {target_device} for inference.")
        model.to(target_device_obj)
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
                if 'cuda' in settings.DEVICE.lower():
                    torch.cuda.empty_cache()

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
                if 'cuda' in settings.DEVICE.lower():
                    torch.cuda.empty_cache()
            
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
            # Always return the tensor on the CPU to prevent VRAM leaks in the cache
            prompt_tokens = prompt_tokens.cpu()

        if settings.FISHSPEECH_OFFLOAD_DECODER_MODEL:
            if 'cuda' in settings.DEVICE.lower():
                logging.debug("Clearing CUDA cache after voice embedding.")
                torch.cuda.empty_cache()

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

            # If cache is enabled, check for an existing voice before creating a new one.
            if self._voice_cache is not None:
                # The token must be a composite of audio and text to be unique
                token_hash = hashlib.sha256()
                token_hash.update(ref_audio_data)
                token_hash.update(params.ref_text.encode('utf-8'))
                token = token_hash.hexdigest()

                if token in self._voice_cache:
                    prompt_tokens, prompt_text = self._voice_cache[token]
                    logging.info(f"Found matching voice in cache. Reusing token: {token}")
                else:
                    logging.info(f"Creating and caching new voice with token: {token}")
                    prompt_tokens = self._create_prompt_tokens_from_data(ref_audio_data)
                    prompt_text = params.ref_text
                    # Cache the tuple of (tokens, text)
                    self._voice_cache[token] = (prompt_tokens, prompt_text)
                
                response_metadata["voice_cache_token"] = token
                generation_kwargs["prompt_tokens"] = prompt_tokens
                generation_kwargs["prompt_text"] = prompt_text
            else:
                # Caching is disabled: create a one-time voice prompt
                logging.info("Voice cache disabled. Creating temporary voice prompt tokens.")
                prompt_tokens = self._create_prompt_tokens_from_data(ref_audio_data)
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

            # Phase 1: Generate semantic codes with T2S model (LLaMA)
            with torch.inference_mode(), \
                model_context(self._decoder_model, self._decoder_offload_device, settings.FISHSPEECH_OFFLOAD_DECODER_MODEL), \
                model_context(self._t2s_model, settings.DEVICE, settings.FISHSPEECH_OFFLOAD_T2S_MODEL):

                logging.info(f"Generating semantic codes for {len(text_chunks)} chunks with T2S model...")
                
                current_t2s_batch_size = settings.FISHSPEECH_T2S_BATCH_SIZE
                while True:
                    try:
                        all_semantic_codes = generate_t2s_batch(
                            model=self._t2s_model,
                            device=settings.DEVICE,
                            batch_size=current_t2s_batch_size,
                            texts=text_chunks,
                            prompt_text=prompt_text,
                            prompt_tokens=prompt_tokens,
                            temperature=params.temperature,
                            top_p=params.top_p,
                            repetition_penalty=params.repetition_penalty,
                            max_new_tokens=params.max_new_tokens,
                        )
                        break # Success
                    except torch.cuda.OutOfMemoryError as e:
                        if 'cuda' not in settings.DEVICE.lower():
                            raise
                        
                        torch.cuda.empty_cache()
                        if current_t2s_batch_size > 1:
                            new_batch_size = current_t2s_batch_size // 2
                            logging.warning(f"T2S model CUDA OOM with batch size {current_t2s_batch_size}. Retrying with {new_batch_size}.")
                            current_t2s_batch_size = new_batch_size
                        else:
                            raise RuntimeError("T2S model ran out of memory even with batch size 1.") from e
                
                # Move semantic codes to CPU immediately after generation
                all_semantic_codes = [code.cpu() if code is not None else None for code in all_semantic_codes]

            if settings.FISHSPEECH_OFFLOAD_T2S_MODEL or settings.FISHSPEECH_OFFLOAD_DECODER_MODEL:
                if 'cuda' in settings.DEVICE.lower():
                    logging.debug("Clearing CUDA cache after T2S and before Decoder.")
                    torch.cuda.empty_cache()

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
            current_decoder_batch_size = settings.FISHSPEECH_DECODER_BATCH_SIZE

            with torch.inference_mode(), \
                model_context(self._t2s_model, self._t2s_offload_device, settings.FISHSPEECH_OFFLOAD_T2S_MODEL), \
                model_context(self._decoder_model, settings.DEVICE, settings.FISHSPEECH_OFFLOAD_DECODER_MODEL):

                logging.info(f"Decoding {len(valid_codes_with_indices)} semantic code sequences in batches of up to {current_decoder_batch_size}.")
                
                i = 0
                while i < len(valid_codes_with_indices):
                    batch_end = i + current_decoder_batch_size
                    batch_of_indexed_codes = valid_codes_with_indices[i:batch_end]
                    
                    if not batch_of_indexed_codes:
                        break

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
                    
                    try:
                        decoded_waveforms_batch, _ = self._decoder_model.decode(batched_codes, feature_lengths=code_lengths)
                        decoded_waveforms_batch = decoded_waveforms_batch.cpu()

                        # Place the decoded waveforms back into the correct positions in the final list
                        for original_idx, waveform in zip(original_indices, decoded_waveforms_batch):
                            final_waveforms[original_idx] = waveform
                        
                        i += len(batch_of_indexed_codes) # Move to the next chunk
                    except torch.cuda.OutOfMemoryError as e:
                        if 'cuda' not in settings.DEVICE.lower():
                            raise

                        torch.cuda.empty_cache()
                        if current_decoder_batch_size > 1:
                            new_batch_size = current_decoder_batch_size // 2
                            logging.warning(
                                f"Decoder model CUDA OOM with batch size {current_decoder_batch_size}. "
                                f"Retrying this batch with size {new_batch_size}."
                            )
                            current_decoder_batch_size = new_batch_size
                        else:
                             raise RuntimeError("Decoder model ran out of memory even with batch size 1.") from e

            return final_waveforms
        except Exception as e:
            # Wrap any unexpected internal error.
            raise EngineExecutionError(f"Fish-Speech model failed during generation: {e}") from e


# Singleton instance for the API to use
fish_speech_engine = FishSpeechEngine()