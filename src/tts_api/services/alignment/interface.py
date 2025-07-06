import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
from pydantic import BaseModel, Field

# --- Standardized Data Models for Alignment Results ---

class WordSegment(BaseModel):
    word: str
    start: Optional[float] = Field(None, description="Start time of the word in seconds.")
    end: Optional[float] = Field(None, description="End time of the word in seconds.")
    score: Optional[float] = Field(None, description="Confidence score of the alignment for this word.")

class AlignmentResult(BaseModel):
    words: List[WordSegment]
    language: str

# --- Abstract Service Interface ---

class AbstractAlignmentService(ABC):
    """
    Defines the contract for any alignment service, ensuring interchangeability.
    """
    @abstractmethod
    def align(self, audio_data: np.ndarray, sample_rate: int, text: str, language: str) -> Optional[AlignmentResult]:
        """
        Performs forced alignment on the given audio and text.

        Args:
            audio_data: A numpy array of the audio waveform.
            sample_rate: The sample rate of the audio.
            text: The ground-truth text to align.
            language: The language code of the text.

        Returns:
            An AlignmentResult object if successful, otherwise None.
        """
        pass

    @abstractmethod
    def align_batch(self, jobs: List[Tuple[np.ndarray, int, str, str]]) -> List[Optional[AlignmentResult]]:
        """
        Performs forced alignment on a batch of audio and text jobs.

        Args:
            jobs: A list of tuples, where each tuple contains:
                - audio_data (np.ndarray): The audio waveform.
                - sample_rate (int): The audio sample rate.
                - text (str): The ground-truth text.
                - language (str): The language code.

        Returns:
            A list of AlignmentResult objects or None, corresponding to each input
            job. The order of the output list MUST match the input list.
        """
        pass