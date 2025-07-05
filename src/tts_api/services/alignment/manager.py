from .api_provider import api_alignment_service_instance as alignment_service_instance

def get_alignment_service():
    """Provides a singleton instance of the configured alignment service."""
    return alignment_service_instance