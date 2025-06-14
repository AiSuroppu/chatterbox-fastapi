from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
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

    def prepare_generation(
        self,
        params: BaseModel,
        ref_audio_data: Optional[bytes]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        An optional, engine-specific pre-processing step before generation.
        This can be used for tasks like voice cloning, caching, etc.

        Args:
            params: The engine-specific Pydantic parameter model.
            ref_audio_data: The raw bytes of the reference audio, if provided.

        Returns:
            A tuple containing:
            - generation_kwargs (Dict): A dictionary of arguments to be passed directly
              to the `generate` method.
            - response_metadata (Dict): A dictionary of metadata to be returned to the
              client (e.g., in response headers).
        """
        # Default implementation for engines that don't need this.
        return {}, {}
