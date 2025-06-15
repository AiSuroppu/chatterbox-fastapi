import pyphen
import logging
from functools import lru_cache
from tts_api.core.config import settings

@lru_cache(maxsize=settings.PYPHEN_CACHE_SIZE)
def get_pyphen(lang: str) -> pyphen.Pyphen:
    """
    Returns a cached pyphen.Pyphen instance for the given language.
    This avoids expensive dictionary loading on every call.
    """
    logging.info(f"Creating and caching new pyphen.Pyphen instance for lang='{lang}'.")
    try:
        return pyphen.Pyphen(lang=lang)
    except KeyError:
        logging.error(f"Pyphen dictionary not found for lang='{lang}'. Falling back to 'en'.")
        # Provide a fallback to prevent total failure if an unsupported lang is requested.
        return pyphen.Pyphen(lang='en_US') # A common fallback