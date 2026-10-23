# NISAR Image Enhancement Scripts

This directory contains scripts for applying image enhancement techniques to NISAR GCOV imagery.

## Available Scripts

- `run_black_tophat.py`: Applies black top-hat enhancement to NISAR image tiles.
- `run_unsharp_mask.py`: Applies unsharp masking to NISAR image tiles.

Both scripts use the shared NISAR preprocessing module (`src/utils/inference_strategies/nisar_preprocessing.py`) to load image tiles, handle invalid pixels, and convert backscatter values to decibels.

## Setup

Ensure the required NISAR GCOV data is available in the following directory structure:

```text
DATA_DIR/
└── nisar_data/
    └── nisar_gcov_pig/
        └── NISAR_L2_PR_GCOV_...h5
```

Set `DATA_DIR` in the project's `.env` file to the directory containing `nisar_data`.

## Running the Scripts

From the project root, run:

```bash
make black-tophat
make unsharp-mask
```

Alternatively, activate the project's virtual environment and configure `DATA_DIR`:

```bash
source .venv/bin/activate
export DATA_DIR="/path/to/your/data"

python3 scripts/run_black_tophat.py
python3 scripts/run_unsharp_mask.py
```

The scripts process predefined NISAR image tiles using their configured enhancement parameters.