import torch
import random
import numpy as np
import io
import logging
from pydantic import BaseModel
from typing import Union, List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum, auto

from tts_api.core.models import BaseTTSRequest, PostProcessingOptions
from tts_api.core.exceptions import ClientRequestError, EngineExecutionError, NoValidCandidatesError
from tts_api.tts_engines.base import AbstractTTSEngine
from tts_api.services.text_processing import process_and_chunk_text, TextChunk, BoundaryType
from tts_api.services.audio_processor import post_process_audio, get_speech_timestamps, calculate_speech_ratio_from_timestamps
from tts_api.services.validation import run_validation_pipeline, ValidationResult
from tts_api.services.alignment.interface import AlignmentResult, WordSegment
from tts_api.services.alignment.manager import get_alignment_service


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
class AudioAnalysis:
    """A container for pre-computed metrics of a generated audio chunk."""
    vad_speech_timestamps: List[Dict[str, int]]
    alignment_result: Optional[AlignmentResult] = None

@dataclass
class GeneratedSegment:
    """Holds the result of a single successful audio generation for a text chunk."""
    waveform: torch.Tensor
    sample_rate: int
    # A single object containing all analysis results for this segment.
    analysis: AudioAnalysis
    # This score can be a VAD ratio or an aggregate alignment score.
    score: float

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
    result: Optional[GeneratedSegment] = None
    last_failure_reason: Optional[str] = None

@dataclass
class SynthesisResult:
    """A container for the final audio and any associated metadata."""
    audio_buffer: io.BytesIO
    metadata: Dict[str, Any] = field(default_factory=dict)
    word_segments: Optional[List[WordSegment]] = None


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

def _trim_waveform_with_vad(segment: GeneratedSegment) -> torch.Tensor:
    """Trims silence from a waveform using its pre-computed VAD timestamps."""
    timestamps = segment.analysis.vad_speech_timestamps
    if not timestamps:
        # If VAD found no speech, return an empty tensor.
        num_channels = segment.waveform.shape[0] if segment.waveform.ndim > 1 else 1
        return torch.zeros((num_channels, 0), dtype=segment.waveform.dtype, device=segment.waveform.device)
    
    start_sample = timestamps[0]['start']
    end_sample = timestamps[-1]['end']
    return segment.waveform[..., start_sample:end_sample]

def _assemble_final_waveform(
    final_segments: List[GeneratedSegment],
    input_segment_info: List[TextChunk],
    post_processing_opts: PostProcessingOptions,
    sample_rate: int
) -> Tuple[torch.Tensor, List[WordSegment]]:
    """Assembles the final audio and calculates final, offset word timestamps."""
    if not final_segments:
        return torch.zeros((1, 0)), []

    final_audio_parts = []
    final_word_segments = []
    current_duration_s = 0.0

    num_channels = final_segments[0].waveform.shape[0] if final_segments[0].waveform.ndim > 1 else 1
    dtype = final_segments[0].waveform.dtype
    device = final_segments[0].waveform.device

    # Handle lead-in silence for the very first segment
    if post_processing_opts.lead_in_silence_ms is not None:
        lead_in_samples = int((post_processing_opts.lead_in_silence_ms / 1000) * sample_rate)
        if lead_in_samples > 0:
            final_audio_parts.append(torch.zeros((num_channels, lead_in_samples), dtype=dtype, device=device))
            current_duration_s += post_processing_opts.lead_in_silence_ms / 1000.0

    for i, segment in enumerate(final_segments):
        waveform_to_process = segment.waveform
        vad_timestamps = segment.analysis.vad_speech_timestamps
        segment_words = segment.analysis.alignment_result.words if segment.analysis.alignment_result else []
        is_first_segment = (i == 0)
        is_last_segment = (i == len(final_segments) - 1)
        
        # If user is controlling lead-in/out, we trim the VAD-detected silence from the edges.
        # Otherwise, if trim_segment_silence is true, we still trim it.
        # If trim_segment_silence is false, we keep the original waveform unless a lead-in/out override forces a trim.
        if is_first_segment and post_processing_opts.lead_in_silence_ms is not None:
            if vad_timestamps: waveform_to_process = waveform_to_process[..., vad_timestamps[0]['start']:]
        
        if is_last_segment and post_processing_opts.lead_out_silence_ms is not None:
            if vad_timestamps: waveform_to_process = waveform_to_process[..., :vad_timestamps[-1]['end']]

        if post_processing_opts.trim_segment_silence:
            if is_first_segment and not is_last_segment:
                if vad_timestamps: waveform_to_process = waveform_to_process[..., :vad_timestamps[-1]['end']]
            elif is_last_segment and not is_first_segment:
                if vad_timestamps: waveform_to_process = waveform_to_process[..., vad_timestamps[0]['start']:]
            elif not is_first_segment and not is_last_segment:
                waveform_to_process = _trim_waveform_with_vad(segment)

        final_audio_parts.append(waveform_to_process)
        segment_duration_s = waveform_to_process.shape[-1] / sample_rate
        
        for word in segment_words:
            if word.start is not None and word.end is not None:
                offset_word = word.model_copy()
                offset_word.start += current_duration_s
                offset_word.end += current_duration_s
                final_word_segments.append(offset_word)

        current_duration_s += segment_duration_s

        # Add inter-segment silence if this is not the last segment
        if not is_last_segment:
            segment_info = input_segment_info[i]
            pause_ms = post_processing_opts.inter_segment_silence_fallback_ms
            if segment_info.boundary_type == BoundaryType.PARAGRAPH:
                pause_ms = post_processing_opts.inter_segment_silence_paragraph_ms
            elif segment_info.boundary_type == BoundaryType.SENTENCE:
                pause_ms = post_processing_opts.inter_segment_silence_sentence_ms

            if pause_ms > 0:
                silence_samples = int((pause_ms / 1000) * sample_rate)
                final_audio_parts.append(torch.zeros((num_channels, silence_samples), dtype=dtype, device=device))
                current_duration_s += pause_ms / 1000.0

    # Handle lead-out silence for the very last segment
    if post_processing_opts.lead_out_silence_ms is not None:
        lead_out_samples = int((post_processing_opts.lead_out_silence_ms / 1000) * sample_rate)
        if lead_out_samples > 0:
            final_audio_parts.append(torch.zeros((num_channels, lead_out_samples), dtype=dtype, device=device))

    final_waveform = torch.cat([p for p in final_audio_parts if p.numel() > 0], dim=1) if final_audio_parts else torch.zeros((num_channels, 0), dtype=dtype, device=device)
    return final_waveform, final_word_segments


def generate_speech_from_request(
    req: BaseTTSRequest,
    engine: AbstractTTSEngine,
    engine_params: BaseModel,
    ref_audio_data: bytes | None = None
) -> SynthesisResult:
    input_segments = process_and_chunk_text(text=req.text, options=req.text_processing)
    if not input_segments:
        raise ClientRequestError("No text to synthesize after processing. The input may be empty or contain only removable characters.")
    
    generation_kwargs, response_metadata = engine.prepare_generation(
        params=engine_params,
        ref_audio_data=ref_audio_data
    )

    logging.info(f"Processing {len(input_segments)} text segments. Generating {req.best_of} candidate(s) per segment.")

    # Phase 1: Initialization
    # Pre-calculate one initial seed for each candidate generation pass.
    # This ensures all segments of a single candidate share the same seed for the first attempt.
    candidate_seeds = [_get_initial_seed(req.seed, j) for j in range(req.best_of)]
    logging.info(f"Initial seeds for {req.best_of} candidate(s): {candidate_seeds}")

    all_jobs = []
    for i, segment_chunk in enumerate(input_segments):
        for j in range(req.best_of):
            initial_seed = candidate_seeds[j]  # Use the pre-calculated seed
            all_jobs.append(GenerationJob(
                segment_idx=i, candidate_idx=j, text=segment_chunk.text, initial_seed=initial_seed
            ))

    # This flag determines if we need to run the expensive alignment step.
    alignment_needed = req.validation.enable_alignment_validation or req.return_timestamps

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
                **generation_kwargs
            )

            # --- Staged processing for batch alignment ---
            
            # Stage 1: Initial analysis and collection for alignment
            analysis_store: Dict[int, Optional[Tuple[torch.Tensor, AudioAnalysis]]] = {}
            alignment_payload: List[Tuple[np.ndarray, int, str, str]] = []
            alignment_target_analysis: List[AudioAnalysis] = []

            for i, job in enumerate(job_group):
                job.attempt_count += 1
                waveform = generated_waveforms[i]
                
                if waveform is None or waveform.numel() == 0:
                    job.last_failure_reason = "Engine returned empty/None output"
                    analysis_store[i] = None # Mark as failed
                    continue
                
                vad_timestamps = get_speech_timestamps(waveform, engine.sample_rate, req.post_processing)
                analysis_data = AudioAnalysis(vad_speech_timestamps=vad_timestamps)
                analysis_store[i] = (waveform, analysis_data)

                if alignment_needed:
                    waveform_mono = torch.mean(waveform, dim=0) if waveform.ndim > 1 else waveform
                    audio_np = waveform_mono.cpu().numpy().astype(np.float32)
                    alignment_payload.append(
                        (audio_np, engine.sample_rate, job.text, req.text_processing.text_language)
                    )
                    # Keep a direct reference to the analysis object we need to update
                    alignment_target_analysis.append(analysis_data)
            
            # Stage 2: Execute alignment for the entire batch in one call
            if alignment_payload:
                alignment_service = get_alignment_service()
                batch_alignment_results = alignment_service.align_batch(alignment_payload)
                for i, result in enumerate(batch_alignment_results):
                    # The lists are parallel, so we can map results back directly
                    alignment_target_analysis[i].alignment_result = result
            
            # Stage 3: Final validation and scoring for the whole group
            for i, job in enumerate(job_group):
                stored_data = analysis_store.get(i)
                if stored_data is None: # Job failed in stage 1
                    continue
                
                waveform, analysis_data = stored_data

                validation_result = run_validation_pipeline(
                    waveform=waveform,
                    sample_rate=engine.sample_rate,
                    text_chunk=job.text,
                    validation_params=req.validation,
                    post_processing_params=req.post_processing,
                    language=req.text_processing.text_language,
                    analysis_data=analysis_data
                )
                
                if validation_result.is_ok:
                    job.status = JobStatus.SUCCESS
                    # SCORING: Use the same analysis data.
                    if req.validation.enable_alignment_validation:
                        if analysis_data.alignment_result and analysis_data.alignment_result.words:
                            word_scores = [w.score for w in analysis_data.alignment_result.words if w.score is not None]
                            score = sum(word_scores) / len(word_scores) if word_scores else 0.0
                        else:
                            score = 0.0
                            logging.warning(f"Alignment validation passed but no alignment data found for scoring. Defaulting score to 0.")
                    else:
                        score = calculate_speech_ratio_from_timestamps(analysis_data.vad_speech_timestamps, waveform.shape[-1])
                    
                    job.result = GeneratedSegment(waveform=waveform, sample_rate=engine.sample_rate, analysis=analysis_data, score=score)
                    job.last_failure_reason = None
                else:
                    job.last_failure_reason = validation_result.reason
                    if job.attempt_count <= req.max_retries:
                        logging.warning(f"Validation failed for chunk (seg={job.segment_idx}, cand={job.candidate_idx}, att={job.attempt_count}, seed={job.current_seed}): {validation_result.reason}. Will retry.")

    # Phase 3: Finalization & Assembly
    final_failures = sum(1 for job in all_jobs if job.status != JobStatus.SUCCESS)
    if final_failures > 0:
        logging.error(f"{final_failures} jobs failed all {req.max_retries+1} attempts and will be excluded.")
    
    successful_jobs = [job for job in all_jobs if job.status == JobStatus.SUCCESS]
    final_segments = []
    for i, segment_info in enumerate(input_segments):
        candidates_for_segment = [job for job in successful_jobs if job.segment_idx == i]
        if not candidates_for_segment:
            # Find a failed job for this segment to extract the last error message
            failed_jobs_for_segment = [
                job for job in all_jobs if job.segment_idx == i and job.status == JobStatus.FAILED_FINAL
            ]
            last_error = "Validation failed, but no specific reason was captured."
            if failed_jobs_for_segment and failed_jobs_for_segment[0].last_failure_reason:
                last_error = failed_jobs_for_segment[0].last_failure_reason
            
            raise NoValidCandidatesError(
                segment_index=i,
                segment_text=segment_info.text,
                last_error=last_error
            )
        
        best_job = max(candidates_for_segment, key=lambda j: j.result.score)
        final_segments.append(best_job.result)
        logging.info(f"Selected best candidate for segment {i+1} (score: {best_job.result.score:.4f}, seed: {best_job.current_seed})")

    if not final_segments:
        raise EngineExecutionError("TTS generation failed to produce any audio.")

    final_waveform, final_word_segments = _assemble_final_waveform(
        final_segments=final_segments,
        input_segment_info=input_segments,
        post_processing_opts=req.post_processing,
        sample_rate=engine.sample_rate
    )

    if final_waveform is None or final_waveform.numel() == 0:
        raise EngineExecutionError("TTS generation failed to produce any final audio after assembly.")

    audio_buffer = post_process_audio(
        final_waveform,
        engine.sample_rate,
        req.post_processing
    )

    return SynthesisResult(audio_buffer=audio_buffer, metadata=response_metadata, word_segments=final_word_segments)