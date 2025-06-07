import torch
import random
import numpy as np
import io
import os
import tempfile
import logging
from typing import Union, List

from tts_api.core.models import BaseTTSRequest
from tts_api.tts_engines.base import AbstractTTSEngine
from tts_api.utils.text_processing import process_and_chunk_text
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
    chunk_idx: int,
    candidate_idx: int,
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
    Main service function to orchestrate TTS generation.
    It processes text into chunks. For each chunk, it generates `best_of` candidates,
    scores them, and picks the best one. The best chunks are then concatenated.
    """
    text_chunks = process_and_chunk_text(
        text=req.text,
        text_options=req.text_processing,
    )

    tmp_ref_file = None
    ref_audio_path = None
    final_waveform = None
    try:
        # Create a temporary file for the reference audio if it exists
        if ref_audio_data:
            tmp_ref_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_ref_file.write(ref_audio_data)
            tmp_ref_file.close()  # Close the file so the engine can access it
            ref_audio_path = tmp_ref_file.name
            logging.info(f"Using reference audio from temporary file: {ref_audio_path}")

        final_waveform_list = []
        num_chunks = len(text_chunks)
        logging.info(f"Processing text into {num_chunks} chunks. Generating {req.best_of} candidate(s) per chunk.")

        for i, chunk in enumerate(text_chunks):
            if not chunk.strip():
                continue

            logging.info(f"Processing chunk {i+1}/{num_chunks}: '{chunk[:50]}...'")
            
            chunk_candidates = []
            for j in range(req.best_of):
                seed = _get_seed_for_generation(
                    req.seed,
                    chunk_idx=i,
                    candidate_idx=j,
                )
                set_seed(seed)
                logging.debug(f"  Generating candidate {j+1}/{req.best_of} for chunk {i+1} with seed {seed}")
                
                candidate_waveform = engine.generate(chunk, engine_params, ref_audio_path=ref_audio_path)

                if candidate_waveform.numel() == 0:
                    logging.warning(f"  Candidate {j+1} for chunk {i+1} resulted in empty audio.")
                    continue

                # Score the candidate chunk
                score = get_speech_ratio(candidate_waveform, engine.sample_rate, req.post_processing)
                chunk_candidates.append({'waveform': candidate_waveform, 'score': score, 'seed': seed})
                logging.debug(f"  Candidate {j+1} score (speech ratio): {score:.4f}")
            
            if not chunk_candidates:
                raise ValueError(f"TTS generation failed to produce any valid candidates for chunk {i+1}: '{chunk[:50]}...'.")

            # Select the best candidate for this chunk
            best_chunk_candidate = max(chunk_candidates, key=lambda c: c['score'])
            final_waveform_list.append(best_chunk_candidate['waveform'])
            logging.info(f"  Selected best candidate for chunk {i+1} with score {best_chunk_candidate['score']:.4f} and seed {best_chunk_candidate['seed']}.")

        if not final_waveform_list:
            raise ValueError("TTS generation failed to produce any audio.")

        # Concatenate the best waveform from each chunk
        final_waveform = torch.cat(final_waveform_list, dim=1)

    finally:
        # Ensure the temporary reference audio file is always cleaned up
        if tmp_ref_file and os.path.exists(tmp_ref_file.name):
            os.remove(tmp_ref_file.name)
            logging.info(f"Cleaned up temporary reference audio file: {tmp_ref_file.name}")

    if final_waveform is None or final_waveform.numel() == 0:
        raise ValueError("TTS generation failed to produce any audio.")

    # Post-process ONLY the complete, final waveform
    audio_buffer = post_process_audio(
        final_waveform,
        engine.sample_rate,
        req.post_processing
    )

    return audio_buffer