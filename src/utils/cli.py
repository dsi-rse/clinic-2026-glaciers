"""CLI for the clinic-2026-glaciers project."""
import json
import logging
from pathlib import Path

import click

from utils.evaluation import discover_evaluators
from utils.inference import discover_inference_strategies
from utils.pipeline import DEFAULT_INPUT, DEFAULT_OUTPUT_DIR, run_evaluation, run_inference, run_pipeline


logger = logging.getLogger(__name__)

STRATEGIES = discover_inference_strategies()
EVALUATORS = discover_evaluators()


def _parse_params(params: tuple[str, ...]) -> dict[str, object]:
    """Convert a tuple of 'key=value' strings into a dict.

    Values are automatically coerced via ``json.loads`` so that numbers,
    booleans, and lists come through with the right types.  Plain strings
    are kept as-is.
    """
    parsed: dict[str, object] = {}
    for p in params:
        key, _, value = p.partition("=")
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        parsed[key] = value
    return parsed


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
def cli() -> None:
    """clinic-2026-glaciers CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--strategy", "strategy_name", required=True,
              type=click.Choice(STRATEGIES.keys(), case_sensitive=True),
              help="Name of the strategy to run.")
@click.option("--param", "params", multiple=True, help="Strategy param as key=value.")
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_INPUT,
    help="Path to input data.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    help="Base output directory.",
)
def infer(strategy_name: str, params: tuple[str, ...], input_path: Path, output_dir: Path) -> None:
    """Run an inference strategy on the input data."""
    run_dir = run_inference(strategy_name, input_path, base_output_dir=output_dir, params=_parse_params(params))
    click.echo(f"Done. Outputs saved to {run_dir}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--evaluator", "evaluator_name", required=True,
              type=click.Choice(EVALUATORS.keys(), case_sensitive=True),
              help="Name of the evaluator to use.")
@click.option(
    "--run-dir",
    "run_dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to an inference run directory.",
)
@click.option(
    "--expected",
    "expected_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to expected/ground-truth outputs.",
)
def evaluate(evaluator_name: str, run_dir: Path, expected_path: Path) -> None:
    """Evaluate inference outputs against expected results."""
    run_evaluation(evaluator_name, run_dir, expected_path)
    click.echo(f"Done. Evaluation results saved to {run_dir}")


# ---------------------------------------------------------------------------
# run (infer + evaluate)
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--strategy", "strategy_name", required=True,
              type=click.Choice(STRATEGIES.keys(), case_sensitive=True),
              help="Name of the strategy to run.")
@click.option("--evaluator", "evaluator_name", required=True,
              type=click.Choice(EVALUATORS.keys(), case_sensitive=True),
              help="Name of the evaluator to use.")
@click.option("--param", "params", multiple=True, help="Strategy param as key=value.")
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    default=DEFAULT_INPUT,
    help="Path to input data.",
)
@click.option(
    "--expected",
    "expected_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to expected/ground-truth outputs.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    help="Base output directory.",
)
def run(
    strategy_name: str,
    evaluator_name: str,
    params: tuple[str, ...],
    input_path: Path,
    expected_path: Path,
    output_dir: Path,
) -> None:
    """Run inference and evaluation in a single step."""
    run_dir = run_pipeline(
        strategy_name, evaluator_name, input_path,
        expected_path=expected_path, base_output_dir=output_dir,
        params=_parse_params(params),
    )
    click.echo(f"Done. Results saved to {run_dir}")
