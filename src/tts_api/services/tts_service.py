import torch
import random
import numpy as np
import io
import os
import tempfile
import logging

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
    logging.info(f"Seed set to: {seed}")

def _generate_single_candidate(
    text_chunks: list[str],
    base_seed: int,
    engine: AbstractTTSEngine,
    engine_params: 'BaseModel',
    ref_audio_path: str | None,
    candidate_index: int = 0
) -> torch.Tensor:
    """Helper function to generate one full audio waveform."""
    if base_seed == 0:
        # Use a different random seed for each candidate
        set_seed(random.randint(1, 2**32 - 1))
    else:
        # Create a deterministic but unique seed for each candidate
        set_seed(base_seed + candidate_index * 1000)

    waveform_list = []
    for i, chunk in enumerate(text_chunks):
        if not chunk.strip():
            continue

        wav_tensor = engine.generate(chunk, engine_params, ref_audio_path=ref_audio_path)
        waveform_list.append(wav_tensor)
    
    if not waveform_list:
        return torch.tensor([])

    return torch.cat(waveform_list, dim=1)

def generate_speech_from_request(
    req: BaseTTSRequest, 
    engine: AbstractTTSEngine, 
    engine_params: 'BaseModel',
    ref_audio_data: bytes | None = None
) -> io.BytesIO:
    """
    Main service function to orchestrate TTS generation.
    If best_of > 1, it generates multiple versions, evaluates them,
    and returns the best one.
    """
    text_chunks = process_and_chunk_text(
        text=req.text,
        text_options=req.text_processing,
    )
    
    tmp_ref_file = None
    ref_audio_path = None
    try:
        # Create a temporary file for the reference audio if it exists
        if ref_audio_data:
            tmp_ref_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_ref_file.write(ref_audio_data)
            tmp_ref_file.close() # Close the file so the engine can access it
            ref_audio_path = tmp_ref_file.name
            logging.info(f"Using reference audio from temporary file: {ref_audio_path}")

        if req.best_of == 1:
            best_waveform = _generate_single_candidate(
                text_chunks, req.seed, engine, engine_params, ref_audio_path
            )
        else:
            logging.info(f"Generating {req.best_of} candidates for evaluation.")
            candidates = []
            for i in range(req.best_of):
                logging.info(f"Generating candidate {i+1}/{req.best_of}...")
                candidate_waveform = _generate_single_candidate(
                    text_chunks, req.seed, engine, engine_params, ref_audio_path, candidate_index=i
                )
                if candidate_waveform.numel() == 0:
                    continue

                # Evaluate the candidate using the speech ratio
                score = get_speech_ratio(candidate_waveform, engine.sample_rate, req.post_processing)
                candidates.append({'waveform': candidate_waveform, 'score': score})
                logging.info(f"Candidate {i+1} score (speech ratio): {score:.4f}")

            if not candidates:
                 raise ValueError("TTS generation failed to produce any valid candidates.")
            
            # Select the candidate with the highest score
            best_candidate = max(candidates, key=lambda c: c['score'])
            best_waveform = best_candidate['waveform']
            logging.info(f"Selected best candidate with score: {best_candidate['score']:.4f}")

    finally:
        # Ensure the temporary reference audio file is always cleaned up
        if tmp_ref_file and os.path.exists(tmp_ref_file.name):
            os.remove(tmp_ref_file.name)
            logging.info(f"Cleaned up temporary reference audio file: {tmp_ref_file.name}")


    if best_waveform is None or best_waveform.numel() == 0:
        raise ValueError("TTS generation failed to produce any audio.")

    # Post-process ONLY the best waveform
    audio_buffer = post_process_audio(
        best_waveform, 
        engine.sample_rate, 
        req.post_processing
    )
    
    return audio_buffer