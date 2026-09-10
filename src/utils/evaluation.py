from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from typing import Any

import matplotlib.pyplot as plt

from utils.register import discover_subclasses, get_subclass


class EvaluationStatus(StrEnum):
    """Why a particular key could not be evaluated."""
    EVALUATED = "evaluated"
    MISSING_FROM_PREDICTED = "missing_from_predicted"
    MISSING_FROM_ACTUAL = "missing_from_actual"


class AbstractEvaluator(ABC):
    @abstractmethod
    def evaluate_single_output(self, predicted: Any, actual: Any) -> dict[str, Any]:
        pass

    def make_plots(self, evaluation_results: dict[str, Any]) -> dict[str, plt.Figure]:
        """Make plots for the evaluation results.

        Returns:
            A dict mapping plot name to matplotlib Figure.
        """
        return {}

    def evaluate_all(self, predicted: Mapping[str, Any], actual: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        """Evaluate all the outputs.

        Args:
            predicted: A dict mapping keys to predicted output values.
            actual: A dict mapping keys to actual/ground-truth output values.

        Returns:
            A dict mapping keys to evaluation result dicts.
        """
        evaluation_results = {}
        for k in set(predicted.keys()) | set(actual.keys()):
            if k not in predicted.keys():
                evaluation_results[k] = {"status": EvaluationStatus.MISSING_FROM_PREDICTED}
            elif k not in actual.keys():
                evaluation_results[k] = {"status": EvaluationStatus.MISSING_FROM_ACTUAL}
            else:
                evaluation_results[k] = {"status": EvaluationStatus.EVALUATED, **self.evaluate_single_output(predicted[k], actual[k])}
        return evaluation_results


def discover_evaluators() -> dict[str, type]:
    """Discover all concrete AbstractEvaluator subclasses in the evaluators package."""
    from utils import evaluators as pkg
    return discover_subclasses(pkg, AbstractEvaluator)


def get_evaluator(name: str) -> type:
    """Get a specific AbstractEvaluator subclass by class name."""
    from utils import evaluators as pkg
    return get_subclass(name, pkg, AbstractEvaluator, label="evaluator")
