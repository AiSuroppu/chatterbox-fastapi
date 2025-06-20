import logging
import hashlib
from typing import List, Optional, Tuple, Dict, Any
from cachetools import LRUCache
from contextlib import contextmanager
import torch
import torch.nn.functional as F

from fish_speech.models.text2semantic.inference import init_model as init_t2s_model, generate_long
from fish_speech.models.dac.inference import load_model as load_decoder_model
from fish_speech.inference_engine.reference_loader import ReferenceLoader

from tts_api.core.config import settings
from tts_api.core.models import FishSpeechParams
from tts_api.core.exceptions import InvalidVoiceTokenException
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
            logging.critical(f"Failed to load FishSpeech models: {e}", exc_info=True)
            raise

    def _create_prompt_tokens_from_data(self, ref_audio_data: bytes) -> torch.Tensor:
        """Encodes reference audio bytes into semantic tokens (the 'voice embedding')."""
        with torch.inference_mode(), \
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
                raise ValueError("Reference audio was provided, but the required 'ref_text' (transcription) is missing in the request.")

            # Ensure T2S model is on CPU while vocoder is active for voice embedding
            with model_context(self._t2s_model, self._t2s_offload_device, settings.FISHSPEECH_OFFLOAD_T2S_MODEL):
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
            raise RuntimeError("Cannot generate, models not loaded.")

        prompt_tokens = kwargs.get("prompt_tokens")
        prompt_text = kwargs.get("prompt_text")

        if prompt_tokens is not None:
            prompt_tokens = prompt_tokens.to(settings.DEVICE)

        # Phase 1: Generate semantic codes with T2S model (LLaMA)
        t2s_batch_size = settings.FISHSPEECH_T2S_BATCH_SIZE
        if t2s_batch_size > 1:
            logging.warning(
                f"FishSpeech T2S batch size is set to {t2s_batch_size}, but the current "
                "implementation processes texts sequentially. This setting is for future-proofing."
            )

        all_semantic_codes = []
        with torch.inference_mode(), \
             model_context(self._t2s_model, settings.DEVICE, settings.FISHSPEECH_OFFLOAD_T2S_MODEL), \
             model_context(self._decoder_model, self._decoder_offload_device, settings.FISHSPEECH_OFFLOAD_DECODER_MODEL):

            logging.info(f"Generating semantic codes for {len(text_chunks)} chunks with T2S model...")
            # This outer loop respects the batch size setting, even if the inner loop is sequential.
            for i in range(0, len(text_chunks), t2s_batch_size):
                batch = text_chunks[i:i + t2s_batch_size]
                # The underlying `generate_long` is sequential, so we still loop here.
                for text in batch:
                    generator = generate_long(
                        model=self._t2s_model,
                        device=settings.DEVICE,
                        decode_one_token=self._t2s_decode_one_token,
                        text=text,
                        prompt_text=prompt_text,
                        prompt_tokens=prompt_tokens,
                        temperature=params.temperature,
                        top_p=params.top_p,
                        repetition_penalty=params.repetition_penalty,
                        max_new_tokens=params.max_new_tokens,
                    )
                    codes_for_chunk = None
                    for response in generator:
                        if response.action == "sample":
                            codes_for_chunk = response.codes.cpu()
                        elif response.action == "next":
                            break
                    if codes_for_chunk is None:
                        logging.error(f"Failed to generate semantic codes for text: '{text[:50]}...'")
                    all_semantic_codes.append(codes_for_chunk)

        # Phase 2: Batch-decode all valid semantic codes with Vocoder
        valid_codes_with_indices = [
            (i, code) for i, code in enumerate(all_semantic_codes) if code is not None
        ]
        if not valid_codes_with_indices:
            logging.warning("No valid semantic codes were generated for any chunk.")
            return [None] * len(text_chunks)
        
        # Sort by length in descending order before batching.
        # This groups tensors of similar lengths, minimizing padding and wasted computation.
        valid_codes_with_indices.sort(key=lambda item: item[1].shape[1], reverse=True)
        
        final_waveforms = [None] * len(text_chunks)
        decoder_batch_size = settings.FISHSPEECH_DECODER_BATCH_SIZE

        # Ensure Decoder is on GPU and T2S is on CPU for this phase
        with torch.inference_mode(), \
             model_context(self._decoder_model, settings.DEVICE, settings.FISHSPEECH_OFFLOAD_DECODER_MODEL), \
             model_context(self._t2s_model, self._t2s_offload_device, settings.FISHSPEECH_OFFLOAD_T2S_MODEL):

            logging.info(f"Decoding {len(valid_codes_with_indices)} semantic code sequences in batches of up to {decoder_batch_size}.")
            
            for i in range(0, len(valid_codes_with_indices), decoder_batch_size):
                batch_of_indexed_codes = valid_codes_with_indices[i:i + decoder_batch_size]
                original_indices = [item[0] for item in batch_of_indexed_codes]
                codes_to_decode = [item[1] for item in batch_of_indexed_codes]
                
                # Pad and batch codes for the current mini-batch.
                # Since the batch is sorted, the first element is the longest.
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


# Singleton instance for the API to use
fish_speech_engine = FishSpeechEngine()