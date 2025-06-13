import torch
import random
import numpy as np
import io
import os
import tempfile
import logging
from typing import Union, List

from tts_api.core.config import settings
from tts_api.core.models import BaseTTSRequest
from tts_api.tts_engines.base import AbstractTTSEngine
from tts_api.services.text_processing import process_and_chunk_text
from tts_api.services.audio_processor import post_process_audio, get_speech_ratio

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    logging.debug(f"Seed set to: {seed}")

def _get_seed_for_generation(
    req_seed: Union[int, List[int]],
    candidate_idx: int,
    chunk_idx: int = 0, # Included for signature compatibility, but unused
) -> int:
    """
    Determines the seed for a specific generation task.

    - If `req_seed` is a list, it uses the seed at `candidate_idx`. This list of
      seeds is reused for each text chunk. If `best_of` exceeds the list length,
      it falls back to random seeds.
    - If `req_seed` is a single integer, it generates a deterministic seed based on
      both the candidate index to ensure variation across the audio.
    - A seed of 0 is always interpreted as "use a random seed".
    """
    if isinstance(req_seed, list):
        # Use the candidate index to pick from the seed list.
        # This list is re-used for every chunk.
        if candidate_idx < len(req_seed):
            seed = req_seed[candidate_idx]
            # A seed of 0 in the list is a special token for "randomize this one"
            return seed if seed != 0 else random.randint(1, 2**32 - 1)
        else:
            # `best_of` is greater than the number of seeds provided.
            # Fallback to random for the remaining candidates.
            if candidate_idx == len(req_seed):  # Log only on the first overflow
                logging.warning(
                    f"best_of ({candidate_idx + 1}+) is greater than the number of seeds provided ({len(req_seed)}). "
                    f"Using random seeds for remaining candidates."
                )
            return random.randint(1, 2**32 - 1)
    else:  # req_seed is an int
        if req_seed == 0:
            # Use a fully random seed for each generation
            return random.randint(1, 2**32 - 1)
        else:
            # Create a deterministic but unique seed for each candidate.
            return req_seed + candidate_idx * 1000

def generate_speech_from_request(
    req: BaseTTSRequest,
    engine: AbstractTTSEngine,
    engine_params: 'BaseModel',
    ref_audio_data: bytes | None = None
) -> io.BytesIO:
    """
    Orchestrates TTS generation, delegating batching to the engine.
    1. Prepares text segments by chunking the input string.
    2. For each of `best_of` candidates:
       - Passes the *entire list* of segments to the engine for generation.
    3. Scores all generated candidates for each segment.
    4. Selects the best candidate for each segment and concatenates them.
    5. Post-processes the final combined audio.
    """
    input_segments = process_and_chunk_text(
        text=req.text, options=req.text_processing
    )

    if not input_segments:
        raise ValueError("No text to synthesize after processing.")

    tmp_ref_file = None
    ref_audio_path = None
    final_waveform = None

    try:
        if ref_audio_data:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_ref_file:
                tmp_ref_file.write(ref_audio_data)
                ref_audio_path = tmp_ref_file.name
            logging.info(f"Using reference audio from temporary file: {ref_audio_path}")

        num_segments = len(input_segments)
        all_candidates = [[] for _ in range(num_segments)]

        logging.info(f"Processing {num_segments} text segments. Generating {req.best_of} candidate(s) per segment.")

        # For each 'best_of' candidate, generate audio for all segments.
        for j in range(req.best_of):
            seed = _get_seed_for_generation(req.seed, candidate_idx=j)
            set_seed(seed)
            logging.debug(f"Generating candidate #{j+1}/{req.best_of} for all segments with seed {seed}")

            # The engine is responsible for its own batching strategy.
            # We pass the full list of segments here.
            generated_waveforms = engine.generate(input_segments, engine_params, ref_audio_path=ref_audio_path)

            if len(generated_waveforms) != num_segments:
                logging.error(f"Engine returned a mismatched number of waveforms. Expected {num_segments}, got {len(generated_waveforms)}. Skipping this candidate.")
                continue

            # Score each generated waveform and store it as a candidate for its respective segment.
            for i, waveform in enumerate(generated_waveforms):
                if waveform.numel() == 0:
                    logging.warning(f"  Candidate for segment {i + 1} was empty.")
                    continue
                
                score = get_speech_ratio(waveform, engine.sample_rate, req.post_processing)
                candidate_data = {'waveform': waveform, 'score': score, 'seed': seed}
                all_candidates[i].append(candidate_data)

        # Select the best candidate for each segment and build the final list of waveforms
        final_waveform_list = []
        for i, segment_text in enumerate(input_segments):
            candidates_for_segment = all_candidates[i]
            if not candidates_for_segment:
                raise ValueError(f"TTS failed to produce any valid candidates for segment {i+1}: '{segment_text[:50]}...'")
            
            best_candidate = max(candidates_for_segment, key=lambda c: c['score'])
            final_waveform_list.append(best_candidate['waveform'])
            logging.info(f"Selected best candidate for segment {i+1} (score: {best_candidate['score']:.4f}, seed: {best_candidate['seed']})")

        if not final_waveform_list:
            raise ValueError("TTS generation failed to produce any audio.")

        # Concatenate the best waveform from each chunk
        final_waveform = torch.cat(final_waveform_list, dim=1)

    finally:
        if ref_audio_path and os.path.exists(ref_audio_path):
            os.remove(ref_audio_path)
            logging.info(f"Cleaned up temporary reference audio file: {ref_audio_path}")

    if final_waveform is None or final_waveform.numel() == 0:
        raise ValueError("TTS generation failed to produce any audio.")

    # Post-process ONLY the complete, final waveform
    audio_buffer = post_process_audio(
        final_waveform,
        engine.sample_rate,
        req.post_processing
    )

    return audio_buffer