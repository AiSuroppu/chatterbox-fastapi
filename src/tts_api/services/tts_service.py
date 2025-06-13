import torch
import random
import numpy as np
import io
import os
import tempfile
import logging
from typing import Union, List
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum, auto

from tts_api.core.models import BaseTTSRequest
from tts_api.tts_engines.base import AbstractTTSEngine
from tts_api.services.text_processing import process_and_chunk_text
from tts_api.services.audio_processor import post_process_audio, get_speech_ratio
from tts_api.services.validation import ALL_VALIDATORS, ValidationResult

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    logging.debug(f"Seed set to: {seed}")

class JobStatus(Enum):
    """Represents the state of a single generation job."""
    PENDING = auto()
    SUCCESS = auto()
    FAILED_FINAL = auto()

@dataclass
class GenerationJob:
    # Identifiers & Input
    segment_idx: int
    candidate_idx: int
    text: str
    initial_seed: int

    # State Tracking
    status: JobStatus = JobStatus.PENDING
    attempt_count: int = 0
    current_seed: int = 0
    
    # Results
    waveform: torch.Tensor | None = None
    score: float = 0.0

def _get_initial_seed(req_seed: Union[int, List[int]], candidate_idx: int) -> int:
    """Determines the seed for the FIRST attempt of a generation job."""
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

        logging.info(f"Processing {len(input_segments)} text segments. Generating {req.best_of} candidate(s) per segment.")

        # Phase 1: Initialization
        # Pre-calculate one initial seed for each candidate generation pass.
        # This ensures all segments of a single candidate share the same seed for the first attempt.
        candidate_seeds = [_get_initial_seed(req.seed, j) for j in range(req.best_of)]
        logging.info(f"Initial seeds for {req.best_of} candidate(s): {candidate_seeds}")

        all_jobs = []
        for i, segment_text in enumerate(input_segments):
            for j in range(req.best_of):
                initial_seed = candidate_seeds[j]  # Use the pre-calculated seed
                all_jobs.append(GenerationJob(
                    segment_idx=i, candidate_idx=j, text=segment_text, initial_seed=initial_seed
                ))

        # Phase 2: Multi-Pass Processing Loop
        for attempt in range(req.max_retries + 1):
            jobs_to_process = [job for job in all_jobs if job.status == JobStatus.PENDING]
            if not jobs_to_process:
                break
            
            logging.info(f"--- Processing attempt {attempt+1}/{req.max_retries+1}: {len(jobs_to_process)} jobs pending ---")

            # Assign seeds for the current processing pass
            if attempt == 0:
                # On the first attempt, each job uses its pre-assigned initial seed.
                for job in jobs_to_process:
                    job.current_seed = job.initial_seed
            else:
                # For retries, generate a new random seed for EACH candidate that has pending jobs.
                # All pending jobs for the SAME candidate will share this new seed, allowing them
                # to be batched efficiently. This ensures different candidates get different seeds.
                candidates_to_retry = sorted(list({job.candidate_idx for job in jobs_to_process}))
                
                # Generate one new seed per candidate needing a retry
                candidate_retry_seeds = {
                    cand_idx: random.randint(1, 2**32 - 1)
                    for cand_idx in candidates_to_retry
                }
                
                if candidate_retry_seeds:
                    logging.info(f"Generated new retry seeds for {len(candidate_retry_seeds)} candidate(s): {candidate_retry_seeds}")
                
                # Assign the new per-candidate seeds to the corresponding jobs
                for job in jobs_to_process:
                    job.current_seed = candidate_retry_seeds[job.candidate_idx]
            
            jobs_by_seed = defaultdict(list)
            for job in jobs_to_process:
                jobs_by_seed[job.current_seed].append(job)

            for seed, job_group in jobs_by_seed.items():
                set_seed(seed)
                texts_to_generate = [job.text for job in job_group]
                
                # The service calls the engine with a list of texts; the engine handles its own batching.
                generated_waveforms = engine.generate(
                    texts_to_generate, 
                    engine_params, 
                    ref_audio_path=ref_audio_path
                )

                for i, job in enumerate(job_group):
                    job.attempt_count += 1
                    waveform = generated_waveforms[i]
                    
                    final_result = ValidationResult(is_ok=True)
                    if waveform is None or waveform.numel() == 0:
                        final_result = ValidationResult(is_ok=False, reason="Engine returned empty/None output")
                    else:
                        for validator in ALL_VALIDATORS:
                            result = validator.is_valid(waveform, engine.sample_rate, job.text, req.validation_params)
                            if not result.is_ok:
                                final_result = result
                                break
                    
                    if final_result.is_ok:
                        job.status = JobStatus.SUCCESS
                        job.waveform = waveform
                        job.score = get_speech_ratio(waveform, engine.sample_rate, req.post_processing)
                    elif job.attempt_count <= req.max_retries:
                        logging.warning(f"Validation failed for chunk (seg={job.segment_idx}, cand={job.candidate_idx}, att={job.attempt_count}, seed={job.current_seed}): {final_result.reason}. Will retry.")

        # Phase 3: Finalization & Assembly
        final_failures = 0
        for job in all_jobs:
            if job.status == JobStatus.PENDING:
                job.status = JobStatus.FAILED_FINAL
                final_failures += 1
        if final_failures > 0:
            logging.error(f"{final_failures} jobs failed all {req.max_retries+1} attempts and will be excluded.")
        
        all_candidates = [[] for _ in range(len(input_segments))]
        for job in all_jobs:
            if job.status == JobStatus.SUCCESS:
                all_candidates[job.segment_idx].append({'waveform': job.waveform, 'score': job.score, 'seed': job.initial_seed})

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