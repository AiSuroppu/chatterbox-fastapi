import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass

from tts_api.core.models import ValidationParams

@dataclass
class ValidationResult:
    """A container for the result of a validation check."""
    is_ok: bool
    reason: str = ""

class AbstractValidator(ABC):
    """Defines the interface for all post-generation validation checks."""
    @abstractmethod
    def is_valid(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        text_chunk: str,
        params: ValidationParams
    ) -> ValidationResult:
        """
        Validates a generated waveform against a specific criterion.
        Returns a ValidationResult indicating success or failure.
        """
        pass


# A central registry of all validators to be run.
ALL_VALIDATORS = []