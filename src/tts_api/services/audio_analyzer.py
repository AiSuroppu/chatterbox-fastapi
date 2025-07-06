import torch
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from pydantic import BaseModel
from dataclasses import dataclass

from tts_api.core.models import PostProcessingOptions, ValidationOptions
from tts_api.services.alignment.interface import AlignmentResult
from tts_api.services.alignment.manager import get_alignment_service
from tts_api.services.audio_processor import get_speech_timestamps

logger = logging.getLogger(__name__)

# --- Data Models ---

@dataclass
class AudioAnalysis:
    """A container for pre-computed metrics of a generated audio chunk."""
    vad_speech_timestamps: List[Dict[str, int]]
    alignment_result: Optional[AlignmentResult] = None

class AnalysisJob(BaseModel):
    """Input for a single audio analysis task."""
    waveform: torch.Tensor
    sample_rate: int
    text: str
    
    class Config:
        arbitrary_types_allowed = True


def analyze_batch(
    jobs: List[AnalysisJob],
    post_processing_opts: PostProcessingOptions,
    alignment_needed: bool,
    text_language: str
) -> List[Optional[AudioAnalysis]]:
    """
    Analyzes a batch of generated audio waveforms, performing VAD and alignment.

    Args:
        jobs: A list of AnalysisJob objects to process.
        post_processing_opts: Configuration for VAD.
        alignment_needed: Flag to signal if alignment is needed.
        text_language: The language of the text for alignment.

    Returns:
        A list of AudioAnalysis objects or None, in the same order as the input jobs.
        Returns None for jobs that could not be processed (e.g., empty waveform).
    """
    if not jobs:
        return []

    analysis_results: List[Optional[AudioAnalysis]] = [None] * len(jobs)
    alignment_payload: List[Tuple[np.ndarray, int, str, str]] = []
    # Keep a parallel list of indices to map alignment results back correctly.
    alignment_target_indices: List[int] = []

    # --- Stage 1: Perform VAD and collect jobs for alignment ---
    for i, job in enumerate(jobs):
        if job.waveform is None or job.waveform.numel() == 0:
            logger.warning("Skipping analysis for an empty waveform.")
            continue
        
        # VAD is fast and runs on every job.
        vad_timestamps = get_speech_timestamps(job.waveform, job.sample_rate, post_processing_opts)
        
        # Initialize the analysis result for this job.
        analysis_results[i] = AudioAnalysis(vad_speech_timestamps=vad_timestamps)

        # If alignment is needed for this request, prepare the payload.
        if alignment_needed:
            waveform_mono = torch.mean(job.waveform, dim=0) if job.waveform.ndim > 1 else job.waveform
            audio_np = waveform_mono.cpu().numpy().astype(np.float32)
            alignment_payload.append(
                (audio_np, job.sample_rate, job.text, text_language)
            )
            alignment_target_indices.append(i)

    # --- Stage 2: Execute batched alignment (if required) ---
    if alignment_payload:
        logger.debug(f"Sending a batch of {len(alignment_payload)} jobs for alignment.")
        alignment_service = get_alignment_service()
        try:
            batch_alignment_results = alignment_service.align_batch(alignment_payload)
            
            # Map results back to the correct AudioAnalysis objects.
            for i, result in enumerate(batch_alignment_results):
                target_index = alignment_target_indices[i]
                if analysis_results[target_index] is not None:
                    analysis_results[target_index].alignment_result = result
        except Exception as e:
            logger.error(f"Batch alignment failed with an unexpected error: {e}", exc_info=True)
            # Failure is logged, but we continue without alignment data.

    return analysis_results