class TTSAPIException(Exception):
    """Base exception for all custom errors in the TTS API."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class ClientRequestError(TTSAPIException):
    """
    Raised when a request is syntactically valid but semantically incorrect
    for the requested operation (e.g., missing a required parameter for a specific engine).
    Should result in a 400 Bad Request.
    """
    pass

class EngineExecutionError(TTSAPIException):
    """
    Raised when a TTS engine fails during the actual generation/processing
    due to an internal error (e.g., model crash, OOM).
    Should result in a 500 Internal Server Error.
    """
    pass

class EngineLoadError(TTSAPIException):
    """
    Raised when a TTS engine fails to load its model during startup.
    This is a critical, server-side configuration or resource error.
    """
    pass

class InvalidVoiceTokenException(TTSAPIException):
    """Raised when a voice_cache_token is provided but is not found in the cache."""
    def __init__(self, token: str):
        message = (
            f"The provided voice_cache_token '{token}' is invalid or has expired. "
            "To generate a new token, please repeat the original request using the full "
            "voice specification (e.g., by sending the 'ref_audio' file or 'voice_prompt' again)."
        )
        super().__init__(message)