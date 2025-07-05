import logging
import httpx
import json
import io
import soundfile as sf
import numpy as np
from typing import Optional

from pydantic import ValidationError

from tts_api.core.config import settings
from tts_api.core.exceptions import EngineExecutionError
from .interface import AbstractAlignmentService, AlignmentResult, WordSegment


logger = logging.getLogger(__name__)

class ApiAlignmentService(AbstractAlignmentService):
    """
    An alignment service implementation that calls an external API endpoint.
    """

    def align(self, audio_data: np.ndarray, sample_rate: int, text: str, language: str) -> Optional[AlignmentResult]:
        if not settings.ALIGNMENT_API_URL:
            logger.error("Alignment requested, but ALIGNMENT_API_URL is not configured on the server.")
            raise EngineExecutionError("The server is not configured for alignment processing.")

        # Prepare the multipart/form-data payload
        # 1. Convert numpy array to in-memory WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        
        # 2. Prepare the JSON parameters part
        params_dict = {"text_content": text, "language": language}
        params_json = json.dumps(params_dict)

        # 3. Structure files and data for the request
        files = {'file': ('audio.wav', buffer, 'audio/wav')}
        data = {'params': params_json}
        headers = {}
        if settings.ALIGNMENT_API_KEY:
            headers['Authorization'] = f'Bearer {settings.ALIGNMENT_API_KEY}'

        try:
            logger.debug(f"Sending alignment request to {settings.ALIGNMENT_API_URL}")
            with httpx.Client() as client:
                response = client.post(
                    settings.ALIGNMENT_API_URL,
                    files=files,
                    data=data,
                    headers=headers,
                    timeout=settings.ALIGNMENT_API_TIMEOUT
                )
                response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses

            response_data = response.json()
            
            # The example API returns a `TranscriptionResponse` model. We need to
            # adapt its `word_segments` field to our internal `AlignmentResult` model.
            # This mapping provides a great decoupling layer.
            word_segments = [WordSegment(**w) for w in response_data.get("word_segments", [])]

            return AlignmentResult(words=word_segments, language=response_data.get("language", language))

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

# Singleton instance for the application to use
api_alignment_service_instance = ApiAlignmentService()