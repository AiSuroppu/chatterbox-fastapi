import logging
import httpx
import json
import io
import soundfile as sf
import numpy as np
from typing import Tuple, List, Optional

from pydantic import BaseModel, ValidationError

from tts_api.core.config import settings
from tts_api.core.exceptions import EngineExecutionError
from .interface import AbstractAlignmentService, AlignmentResult, WordSegment


logger = logging.getLogger(__name__)

class ApiAlignmentJob(BaseModel):
    filename: str
    text_content: str

class ApiForcedAlignmentRequest(BaseModel):
    jobs: List[ApiAlignmentJob]
    language: str

class ApiAlignmentService(AbstractAlignmentService):
    """
    An alignment service implementation that calls an external API endpoint.
    Now supports efficient batch processing.
    """

    def align(self, audio_data: np.ndarray, sample_rate: int, text: str, language: str) -> Optional[AlignmentResult]:
        batch_result = self.align_batch([(audio_data, sample_rate, text, language)])
        return batch_result[0] if batch_result else None

    def align_batch(self, jobs: List[Tuple[np.ndarray, int, str, str]]) -> List[Optional[AlignmentResult]]:
        if not settings.ALIGNMENT_API_URL:
            logger.error("Alignment requested, but ALIGNMENT_API_URL is not configured.")
            raise EngineExecutionError("The server is not configured for alignment processing.")
        
        if not jobs:
            return []

        # The API expects one language per request. We enforce it here.
        language = jobs[0][3]
        if any(j[3] != language for j in jobs):
            logger.error("All jobs in a batch must have the same language for API alignment.")
            return [None] * len(jobs) # Soft fail the entire batch

        # 1. Prepare the JSON part of the request
        api_jobs = [
            ApiAlignmentJob(filename=f"job_{i}.wav", text_content=job[2])
            for i, job in enumerate(jobs)
        ]
        request_model = ApiForcedAlignmentRequest(jobs=api_jobs, language=language)
        request_data_json = request_model.model_dump_json()

        # 2. Prepare the files part of the request
        files_payload = []
        # Keep buffers in a list to ensure they are not garbage collected before the request is sent
        buffers = [] 
        for i, (audio_data, sample_rate, _, _) in enumerate(jobs):
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
            buffer.seek(0)
            buffers.append(buffer)
            # The filename here MUST match the filename in the JSON part
            files_payload.append(('audio_files', (f"job_{i}.wav", buffer, 'audio/wav')))

        data_payload = {'request_data': request_data_json}
        headers = {}
        if settings.ALIGNMENT_API_KEY:
            headers['Authorization'] = f'Bearer {settings.ALIGNMENT_API_KEY}'

        try:
            logger.debug(f"Sending batch alignment request with {len(jobs)} jobs to {settings.ALIGNMENT_API_URL}")
            with httpx.Client() as client:
                response = client.post(
                    settings.ALIGNMENT_API_URL,
                    files=files_payload,
                    data=data_payload,
                    headers=headers,
                    timeout=settings.ALIGNMENT_API_TIMEOUT * len(jobs) # Increase timeout for batches
                )
                response.raise_for_status()

            response_data = response.json()
            results_dict = response_data.get('results', {})
            
            # 3. Map the dictionary response back to an ordered list
            final_results = []
            for i, job_spec in enumerate(api_jobs):
                job_result = results_dict.get(job_spec.filename)
                if job_result and job_result.get('status').upper() == 'SUCCESS':
                    data = job_result.get('data', {})
                    word_segments = [WordSegment(**w) for w in data.get("word_segments", [])]
                    lang_code = data.get("language_code", language)
                    final_results.append(AlignmentResult(words=word_segments, language=lang_code))
                else:
                    error_msg = job_result.get('message', 'Unknown error') if job_result else 'No result in response'
                    logger.warning(f"Alignment for job {i} failed: {error_msg}")
                    final_results.append(None)
            
            return final_results

        except httpx.HTTPStatusError as e:
            # If we get a 401/403, it's a server configuration issue. Raise a 500-level error.
            if e.response.status_code in [401, 403]:
                logger.error(
                    f"Authentication with alignment API failed (status {e.response.status_code}). "
                    "This is a server configuration issue."
                )
                raise EngineExecutionError("The server is not properly configured for alignment processing.")
            else:
                # For other errors (e.g., 400 Bad Request), it might be a transient issue with the
                # specific audio/text. We "soft fail" by returning None to allow for retries.
                logger.error(f"Alignment API returned a non-auth error: {e.response.status_code} - {e.response.text}")
                return None
        except httpx.RequestError as e:
            # Network-level errors. The service might be temporarily down. Soft fail.
            logger.error(f"Alignment API request failed: Network error connecting to {e.request.url}. Details: {e}")
            return None
        except (json.JSONDecodeError, KeyError, ValidationError) as e:
            # The API returned a malformed response. Soft fail.
            logger.error(f"Failed to parse or validate response from alignment API: {e}")
            return None
        except Exception as e:
            # Catchall for any other unexpected errors.
            logger.error(f"An unexpected error occurred during API alignment: {e}", exc_info=True)
            return None
        finally:
            for buffer in buffers:
                buffer.close()

# Singleton instance for the application to use
api_alignment_service_instance = ApiAlignmentService()