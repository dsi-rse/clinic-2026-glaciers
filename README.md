# clinic-2026-glaciers

## Project Background

[Please add project background]

## Project Goals

[Please add project background]

## Usage

### Docker

### Docker & Make

We use `docker` and `make` to run our code. There are three built-in `make` commands:

* `make build-only`: This will build the image only. It is useful for testing and making changes to the Dockerfile.
* `make run-notebooks`: This will run a jupyter server which also mounts the current directory into `\program`.
* `make run-interactive`: This will create a container (with the current directory mounted as `\program`) and loads an interactive session. 

The file `Makefile` contains information about about the specific commands that are run using when calling each `make` statement.

### Developing inside a container with VS Code

If you prefer to develop inside a container with VS Code then do the following steps. Note that this works with both regular scripts as well as jupyter notebooks.

1. Open the repository in VS Code
2. At the bottom right a window may appear that says `Folder contains a Dev Container configuration file...`. If it does, select, `Reopen in Container` and you are done. Otherwise proceed to next step. 
3. Click the blue or green rectangle in the bottom left of VS code (should say something like `><` or `>< WSL`). Options should appear in the top center of your screen. Select `Reopen in Container`.





## Working with This Project

The project is built around two ideas:

- **Inference strategies** — different approaches to solving the problem. You will spend most of your time adding and improving these.
- **Evaluators** — code that scores how well a strategy did. You may also add or improve these.

The key files are:

| File/Folder | What it's for |
|---|---|
| `src/utils/inference_strategies/` | **Add your strategies here** |
| `src/utils/evaluators/` | Add or improve evaluators here |
| `src/utils/io.py` | How data is loaded and saved (fill this in for your project) |

---

### How to add an inference strategy

1. Create a new `.py` file in `src/utils/inference_strategies/`. Name it something descriptive, e.g. `my_strategy.py`.
2. Copy this template into the file and fill in your logic:

```python
from typing import Any
from utils.inference import InferenceStrategy

class MyStrategy(InferenceStrategy):
    """One sentence describing what this strategy does."""

    def do_inference(self, inference_input: Any) -> dict[str, Any]:
        # Write your logic here.
        # inference_input is one item from your dataset.
        # Return a dict with your results, e.g.:
        return {"label": "my_prediction"}
```

3. Save the file. Your strategy is now available to use everywhere by its class name — `"MyStrategy"` in this example.

If your strategy has settings you want to tune (like a threshold or a model name), add them as parameters to `__init__`:

```python
def __init__(self, threshold: float = 0.5):
    self.threshold = threshold
```

---

### How to test a strategy on one value in a notebook

Import your strategy class directly and call `do_inference` on a single input:

```python
from utils.inference_strategies.my_strategy import MyStrategy

strategy = MyStrategy(threshold=0.8)
result = strategy.do_inference(my_single_input)
print(result)
```

This is the fastest way to check that your logic works before running it on the full dataset.

---

### How to run the full pipeline in a notebook

Once your strategy works on a single input, run it across the whole dataset and evaluate:

```python
from utils.pipeline import run_pipeline

run_dir = run_pipeline(
    "MyStrategy",
    "ClassifierEvaluator",
    params={"threshold": 0.8},  # optional — only if your strategy takes parameters
    expected_path=...,          # path to the ground-truth data
)
print("Results saved to:", run_dir)
```

Results (outputs, scores, and any plots) are saved to a folder inside `data/output/`.

To run inference and evaluation as separate steps:

```python
from utils.pipeline import run_inference, run_evaluation

run_dir = run_inference("MyStrategy", params={"threshold": 0.8})
run_evaluation("ClassifierEvaluator", run_dir=run_dir, expected_path=...)
```

---

### How to run the pipeline from the command line

From the project root, run inference and evaluation together:

```bash
clinic-2026-glaciers run \
    --strategy MyStrategy \
    --evaluator ClassifierEvaluator \
    --param threshold=0.8 \
    --expected data/expected
```

To run just inference:

```bash
clinic-2026-glaciers infer --strategy MyStrategy --param threshold=0.8
```

To evaluate an existing set of outputs:

```bash
clinic-2026-glaciers evaluate \
    --evaluator ClassifierEvaluator \
    --run-dir data/output/MyStrategy/2025-01-01_12-00-00 \
    --expected data/expected
```

Add `--help` to see all available options for any command:

```bash
clinic-2026-glaciers run --help
```

---

### How to add or improve an evaluator

Evaluators live in `src/utils/evaluators/`. Each evaluator scores a single prediction against the correct answer.

1. Create a new `.py` file in the `evaluators/` folder.
2. Use this template:

```python
from typing import Any
from utils.evaluation import AbstractEvaluator

class MyEvaluator(AbstractEvaluator):
    """One sentence describing what this evaluator measures."""

    def evaluate_single_output(self, predicted: Any, actual: Any) -> dict[str, Any]:
        # Compare predicted to actual. Return a dict of scores.
        return {
            "is_correct": predicted == actual,
        }
```

To add a plot to the evaluation report, add a `make_plots` method. It should return a dict where each key is a plot name and each value is a matplotlib figure — the figures are saved automatically when evaluation runs.

```python
import matplotlib.pyplot as plt

def make_plots(self, evaluation_results: dict) -> dict[str, plt.Figure]:
    fig, ax = plt.subplots()
    # ... build your plot using evaluation_results ...
    return {"my_plot": fig}
```

See `classifier_evaluator.py` for a complete example.


## Style
We use [`ruff`](https://docs.astral.sh/ruff/) to enforce style standards and grade code quality. This is an automated code checker that looks for specific issues in the code that need to be fixed to make it readable and consistent with common standards. `ruff` is run before each commit via [`pre-commit`](https://pre-commit.com/). If it fails, the commit will be blocked and the user will be shown what needs to be changed.

To check for errors locally, first ensure that `pre-commit` is installed by running `pip install pre-commit` followed by `pre-commit install`. Once installed, check for errors by running:
```
pre-commit run --all-files
```

## Repository Structure

### utils
Project python code

### notebooks
Contains short, clean notebooks to demonstrate analysis.

### data

Contains details of acquiring all raw data used in repository. If data is small (<50MB) then it is okay to save it to the repo, making sure to clearly document how to the data is obtained.

If the data is larger than 50MB than you should not add it to the repo and instead document how to get the data in the README.md file in the data directory. 

This [README.md file](/data/README.md) should be kept up to date.

### output
Should contain work product generated by the analysis. Keep in mind that results should (generally) be excluded from the git repository.
