class TTSAPIException(Exception):
    """Base exception for all custom errors in the TTS API."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class InvalidVoiceTokenException(TTSAPIException):
    """Raised when a voice_cache_token is provided but is not found in the cache."""
    def __init__(self, token: str):
        message = (
            f"The provided voice_cache_token '{token}' is invalid or has expired. "
            "To generate a new token, please repeat the original request using the full "
            "voice specification (e.g., by sending the 'ref_audio' file or 'voice_prompt' again)."
        )
        super().__init__(message)