"""Example evaluator that checks for exact match between predicted and actual values."""
from typing import Any

from utils.evaluation import AbstractEvaluator


class ExampleEvaluator(AbstractEvaluator):
    """A simple evaluator that checks whether predicted and actual values match exactly."""

    def evaluate_single_output(self, predicted: Any, actual: Any) -> dict[str, Any]:
        """Compare predicted and actual values for equality.

        Args:
            predicted: The predicted value.
            actual: The actual (ground-truth) value.

        Returns:
            A dict with whether the predicted and actual values match.
        """
        return {
            "is_correct": predicted == actual,
        }
