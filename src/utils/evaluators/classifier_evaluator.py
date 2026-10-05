"""Evaluator for binary or multi-class classification tasks."""

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.evaluation import AbstractEvaluator, EvaluationStatus


class ClassifierEvaluator(AbstractEvaluator):
    """Evaluator for classification tasks. Produces a confusion matrix plot."""

    def evaluate_single_output(self, predicted: Any, actual: Any) -> dict[str, Any]:  # noqa: ANN401
        """Compare a single predicted label against the actual label."""
        return {
            "predicted": predicted,
            "actual": actual,
            "is_correct": predicted == actual,
        }

    def make_plots(self, evaluation_results: dict[str, Any]) -> dict[str, plt.Figure]:
        """Return a confusion matrix heatmap for all evaluated outputs."""
        evaluated = [
            v
            for v in evaluation_results.values()
            if v["status"] == EvaluationStatus.INCLUDED
        ]
        if not evaluated:
            return {}

        pairs = pd.DataFrame(evaluated)
        matrix = pairs.pivot_table(
            index="actual", columns="predicted", aggfunc="size", fill_value=0
        )

        fig, ax = plt.subplots()
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        return {"confusion_matrix": fig}
