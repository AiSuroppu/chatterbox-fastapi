from abc import ABC, abstractmethod
import torch
from pydantic import BaseModel

class AbstractTTSEngine(ABC):
    """Abstract Base Class for Text-to-Speech engines."""

    @abstractmethod
    def load_model(self):
        """Load the TTS model into memory."""
        pass

    @abstractmethod
    def generate(self, text: str, params: BaseModel, **kwargs) -> torch.Tensor:
        """
        Generate audio from text.

        Args:
            text (str): The input text.
            params (BaseModel): Engine-specific parameters.
            **kwargs: Additional engine-specific arguments (e.g., file paths).
        """
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Return the sample rate of the engine's output."""
        pass