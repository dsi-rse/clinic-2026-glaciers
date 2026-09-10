from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_inputs(input_path: Path) -> Mapping[str, Any]:
    """Load inputs from the given path.

    Returns a dict mapping item keys to their input values.
    """
    raise NotImplementedError  # TODO: Implement


def save_outputs(outputs: Mapping[str, Any], output_path: Path) -> None:
    """Save outputs to the given path."""
    raise NotImplementedError  # TODO: Implement


def load_outputs(output_path: Path) -> Mapping[str, Any]:
    """Load outputs from the given path.

    Used for loading both predicted outputs (from an inference run) and
    expected/ground-truth outputs (for evaluation).  Returns a dict mapping
    item keys to their output values.
    """
    raise NotImplementedError  # TODO: Implement

