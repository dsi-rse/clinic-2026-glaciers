# Tutorial: Building and Running Your First Strategy

This walkthrough takes you from zero to a working end-to-end run.
By the end you will have:

1. Started a Docker container and attached VS Code to it
2. Written a new strategy that processes a single input
3. Tested it interactively in a notebook
4. Run it across the full dataset and inspected the evaluation results in a notebook
5. Run the same thing from the command line

Before you start, make sure you've completed the
[computer setup guide](https://github.com/dsi-clinic/the-clinic/blob/main/tutorials/clinic-computer-setup.md)
and have Docker, VS Code, and Make working on your machine. You should also
have
[Box Drive](https://www.box.com/resources/downloads) installed, signed in
with your UChicago account, and syncing (your mentor will have shared the
data folder with you).

For general clinic expectations around code quality, documentation, and
repository standards, see the
[coding standards](https://github.com/dsi-clinic/the-clinic/blob/main/coding-standards/coding-standards.md).

---

## Step 1: Start the Docker Container

All of your work — notebooks and command-line runs — happens inside a Docker
container. This keeps everyone on the team using the exact same environment
regardless of what operating system they're on.

Open a terminal, `cd` into the project directory, and run:

```bash
make run-interactive
```

This builds the Docker image (if it hasn't been built yet) and drops you into
a bash shell inside the container. Leave this terminal open — you'll come back to it for command-line runs later.

### Verify the install

Inside the container, run:

```bash
clinic-2026-glaciers --help
```

You should see a list of available commands. If you get a "command not found"
error, ask your mentor for help.

> **Troubleshooting.** If `make run-interactive` fails, check that Docker
> Desktop is running and that your terminal is a Unix shell (Terminal on Mac,
> WSL on Windows — *not* PowerShell). See the clinic
> [Docker FAQ](https://github.com/dsi-clinic/the-clinic/blob/main/tutorials/Docker.md)
> for common issues.

---

## Step 2: Attach VS Code to the Container

You'll write code and run notebooks from inside VS Code, attached to the
same container you just started.

1. Open VS Code
2. Open the Command Palette (`Cmd+Shift+P` on Mac, `Ctrl+Shift+P` on
   Windows/Linux)
3. Type **Dev Containers: Attach to Running Container** and select it
4. Pick the container from the list (it will have `clinic-2026-glaciers`
   in the name)

VS Code will open a new window connected to the container. Use
**File → Open Folder** and open `/program` — that's where the project lives
inside the container.

> **Tip:** You only need to do this once per session. As long as the container
> is running, VS Code stays attached. If you close VS Code and reopen it
> later, just attach again.

---

## Step 3: Configure Your Data Directory

Before you write any code, make sure the project knows where your shared data
folder lives.

In the VS Code window attached to the container:

1. Copy `.env.example` to `.env`
2. Open `.env` and check the `DATA_DIR` value
3. Make sure it points to the Box folder your mentor shared with you

You can sanity-check that the data is accessible from inside the container:

```bash
ls "$DATA_DIR"
```

If that path is empty or missing, fix it now. Most notebook and CLI issues in
this scaffold come from the data directory being misconfigured.

---

## Step 4: Explore One Real Input First

Before writing a strategy, look at an actual input so you understand the data
shape you need to handle. Create a notebook in `notebooks/` and inspect a
single example.

```python
from utils.settings import DATA_DIR
from utils.io import load_inputs

inputs = load_inputs(DATA_DIR / "input")
key, single_input = next(iter(inputs.items()))

print(key)
single_input
```

Spend a minute answering basic questions before coding:

- What Python type is each input?
- What fields or structure does it contain?
- What does a reasonable output dict need to capture?

Once you understand one real example, writing `do_inference` gets much easier.

---

## Step 5: Create a New Strategy

Every strategy lives in its own file inside
`src/utils/inference_strategies/`.
The framework discovers new strategies automatically — all you have to do is
drop a file in that folder.

In the VS Code window attached to the container, create a new file — for this
tutorial we'll call it `my_strategy.py`:

```
src/utils/inference_strategies/my_strategy.py
```

Paste in this starter template:

```python
"""Strategy that does XYZ."""

from typing import Any

from utils.inference import InferenceStrategy


class MyStrategy(InferenceStrategy):
    """A short description of what this strategy does."""

    def do_inference(self, inference_input: Any) -> dict[str, Any]:
        """Process a single input and return results.

        Args:
            inference_input: One item from the dataset.

        Returns:
            A dict containing the results.
        """
        return {"result": "placeholder"}
```

That's a complete, working strategy. The only method you *must* implement is
`do_inference`. It receives one input and returns a dict with your results.

### Adding configurable parameters

If your strategy has settings you want to experiment with (a threshold, a
model name, a window size, etc.), accept them in `__init__`:

```python
class MyStrategy(InferenceStrategy):
    """Strategy with a configurable threshold."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def do_inference(self, inference_input: Any) -> dict[str, Any]:
        """Process a single input and return results.

        Args:
            inference_input: One item from the dataset.

        Returns:
            A dict containing the results.
        """
        # use self.threshold in your logic
        ...
```

Parameters you set on `self` are automatically recorded in the run metadata,
so you can always look back and see what settings produced a given set of
outputs.

### Check your code style

Before moving on, run the linter to make sure your code passes the project's
style checks:

```bash
pre-commit run --all-files
```

Fix any issues it reports. This same check runs automatically every time you
try to commit, so it's easier to fix things as you go.

---

## Step 6: Test It on a Single Input in a Notebook

Before running anything on the full dataset, make sure your logic works on
one item. In the VS Code window attached to the container, create a new
notebook in the `notebooks/` folder.

> **Note:** When VS Code asks you to select a kernel for the notebook, pick
> the Python interpreter that's already installed in the container. There
> should be only one option.

### Notebook guidelines

Notebooks in this project follow
[clinic standards](https://github.com/dsi-clinic/the-clinic/blob/main/coding-standards/coding-standards.md):

- Start with a **markdown cell** containing a title, your name, the date, and
  a brief description of what the notebook does.
- Keep each code cell to **10 lines or fewer**. If a cell is getting long,
  break it up.
- Keep the total notebook to **under 10 cells**. Notebooks are for
  *demonstrating* results, not for developing large amounts of logic.
- **Do not define functions in notebooks.** All reusable logic belongs in
  `.py` files under `src/`. Import it instead.
- **Do not use `! pip install ...`** in notebooks. All dependencies are
  managed in `pyproject.toml` and installed in the Docker image.
- Put all `import` statements in the **first code cell**.
- Remove any scratch/testing cells before you're done.

### Try it

In your first code cell, enable automatic reloading and put your imports:

```python
%load_ext autoreload
%autoreload 2

from utils.inference_strategies.my_strategy import MyStrategy
from utils.io import load_inputs
from utils.settings import DATA_DIR
```

`%autoreload 2` tells the notebook to re-read your `.py` files every time you
run a cell. That way, when you edit your strategy code and come back to the
notebook, you can just re-run the cell — no need to restart the kernel.

In the next cell, load one input and test your strategy:

```python
inputs = load_inputs(DATA_DIR / "input")
key, single_input = next(iter(inputs.items()))

strategy = MyStrategy()          # pass parameters here if your strategy takes any
result = strategy.do_inference(single_input)
print(key, result)
```

This is the fastest feedback loop you have — use it often. Edit your strategy,
save the file, re-run the cell, and see the updated results immediately.

> **If things get weird,** restart the kernel and re-run all cells. Autoreload
> handles most changes, but some (like renaming a class or changing
> inheritance) need a fresh kernel.

### What to check

- Does `result` have the keys you expect?
- Do the values look reasonable for this input?
- Does it run without errors?

Once you're happy with the output on a handful of individual inputs, move on
to running the full pipeline.

---

## Step 7: Run the Full Pipeline in a Notebook

Now you'll run your strategy across *every* input and evaluate the results.

In the same notebook (or a new one in the attached VS Code), run:

```python
from utils.pipeline import run_pipeline

run_dir = run_pipeline(
    "MyStrategy",                    # the class name of your strategy
    "ExampleEvaluator",              # the evaluator to use (ask your mentor which one)
    expected_path="data/expected",   # path to the correct answers
)
print("Results saved to:", run_dir)
```

If your strategy takes parameters:

```python
run_dir = run_pipeline(
    "MyStrategy",
    "ExampleEvaluator",
    expected_path="data/expected",
    params={"threshold": 0.8},
)
```

### What just happened?

`run_pipeline` does two things in sequence:

1. **Inference** — loads every input, runs `do_inference` on each one, and
   saves all the outputs to a timestamped folder inside `data/output/`.
2. **Evaluation** — compares your outputs to the correct answers and saves
   scores (and any plots) into the same folder.

### Inspecting the results

The returned `run_dir` is a `Path` pointing to the output folder. You can
load and explore the evaluation results right in the notebook:

```python
import json

with open(run_dir / "evaluation.json") as f:
    results = json.load(f)

# Pretty-print the first few results
for key, value in list(results.items())[:5]:
    print(key, value)
```

If you want a more tabular view, convert the results to a DataFrame:

```python
import pandas as pd

results_df = pd.read_json(run_dir / "evaluation.json").T
results_df.head()
```

`pd.DataFrame.from_dict(results, orient="index")` works too if you already
have the JSON loaded into memory.

If the evaluator produces plots (like a confusion matrix), they are saved as
`.png` files in the same folder. You can display them in the notebook:

```python
from IPython.display import Image

Image(filename=str(run_dir / "confusion_matrix.png"))
```

### Running inference and evaluation separately

Sometimes you want to run inference once and then try different evaluators on
the same outputs, or re-evaluate without re-running inference. You can split
the two steps:

```python
from utils.pipeline import run_inference, run_evaluation

# Run inference only
run_dir = run_inference("MyStrategy", params={"threshold": 0.8})

run_evaluation("ExampleEvaluator", run_dir=run_dir, expected_path="data/expected")
```

---

## Step 8: Run the Pipeline from the Command Line

Switch back to the terminal where you ran `make run-interactive` — you should
still have a bash prompt inside the container.

> **Tip:** If you closed that terminal, just run `make run-interactive` again
> from the project directory to start a new session.

Run inference and evaluation together:

```bash
clinic-2026-glaciers run \
    --strategy MyStrategy \
    --evaluator ExampleEvaluator \
    --expected data/expected
```

With parameters:

```bash
clinic-2026-glaciers run \
    --strategy MyStrategy \
    --evaluator ExampleEvaluator \
    --expected data/expected \
    --param threshold=0.8
```

You can also run inference and evaluation as separate commands:

```bash
# Run inference only
clinic-2026-glaciers infer --strategy MyStrategy --param threshold=0.8

# Evaluate an existing run (use the path printed by the infer command)
clinic-2026-glaciers evaluate \
    --evaluator ExampleEvaluator \
    --run-dir data/output/MyStrategy/2025-01-15_14-30-00 \
    --expected data/expected
```

To see all options for any command, add `--help`:

```bash
clinic-2026-glaciers run --help
```

---

## Quick Reference

### Code quality checklist

- [ ] Every file, class, and function has a docstring
  ([Google style](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods))
- [ ] There is no commented-out code
- [ ] `pre-commit run --all-files` passes
- [ ] Notebooks have under 10 cells, each 10 lines or fewer
- [ ] Notebooks don't define functions — those belong in `src/`
- [ ] No `! pip install` in notebooks

### Useful links

- [Clinic coding standards](https://github.com/dsi-clinic/the-clinic/blob/main/coding-standards/coding-standards.md)
- [Computer setup guide](https://github.com/dsi-clinic/the-clinic/blob/main/tutorials/clinic-computer-setup.md)
- [Docker FAQ](https://github.com/dsi-clinic/the-clinic/blob/main/tutorials/Docker.md)
- [Well-documented code example](https://github.com/dsi-clinic/the-clinic/blob/main/coding-standards/code-example.md)

---

## What to Do Next

- **Iterate on your strategy.** Change the logic in `do_inference`, re-test on
  single inputs in a notebook, then re-run the full pipeline to see if your
  scores improve.
- **Try different parameters.** If your strategy accepts parameters, experiment
  with different values and compare the evaluation results across runs.
  Each run gets its own timestamped folder, so nothing is overwritten.
- **Look at the example code.** The files `example_strategy.py` and
  `classifier_evaluator.py` in the source tree are short, readable references.
- **Ask your mentor** if you're not sure what evaluator to use, what the correct
  answers should look like, or how to interpret the evaluation results.
