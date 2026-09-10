"""Basic scaffold for any inference strategy."""
import logging
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from utils.register import discover_subclasses, get_subclass

logger = logging.getLogger(__name__)


class InferenceStatus(StrEnum):
    """Status of a single inference run."""

    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"


class InferenceStrategy(ABC):
    """Abstract base class for inference strategies."""

    @abstractmethod
    def do_inference(self, inference_input: Any) -> dict[str, Any] | None:
        pass

    def do_inference_safe(self, inference_input: Any) -> tuple[dict[str, Any] | None, InferenceStatus]:
        """Run ``do_inference`` with exception handling.

        Returns:
            A tuple of ``(result, status)``.
        """
        try:
            result = self.do_inference(inference_input)
        except Exception:
            logger.exception("Inference failed")
            return None, InferenceStatus.ERROR

        if result is None:
            return None, InferenceStatus.FAILURE

        return result, InferenceStatus.SUCCESS

    def metadata(self) -> dict[str, Any]:
        """Metadata about the inference strategy."""
        return {
            "name": self.__class__.__name__,
            "description": self.__doc__,
            "parameters": self.parameters(),
        }

    def parameters(self) -> dict[str, Any]:
        """Parameters for the inference strategy."""
        return {
            key: value
            for key, value in self.__dict__.items()
            if not callable(value) and not key.startswith('_')
        }


def discover_inference_strategies() -> dict[str, type]:
    """Discover all concrete InferenceStrategy subclasses in the inference_strategies package."""
    from utils import inference_strategies as pkg
    return discover_subclasses(pkg, InferenceStrategy)


def get_inference_strategy(name: str) -> type:
    """Get a specific InferenceStrategy subclass by class name."""
    from utils import inference_strategies as pkg
    return get_subclass(name, pkg, InferenceStrategy, label="inference strategy")
