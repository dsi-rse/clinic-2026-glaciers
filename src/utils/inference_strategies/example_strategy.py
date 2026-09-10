"""Example inference strategy that returns a fixed result."""
from typing import Any

from utils.inference import InferenceStrategy


class ExampleStrategy(InferenceStrategy):
    """A simple example strategy that echoes the input back."""

    def do_inference(self, inference_input: Any) -> dict[str, Any]:
        """Return a fixed result.

        Args:
            inference_input: Any input value.

        Returns:
            A dict with the result.
        """
        return {
            "result": "Hello, world!",
        }
