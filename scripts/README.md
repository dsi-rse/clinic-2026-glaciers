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

## Full-frame enhancement: `enhance.py`

This processes one complete NISAR GCOV HDF5 frame in chunks, using the same
filters as the tile scripts. No model weights or extra downloads are needed.
The script imports `src/utils/inference_strategies/full_frame_enhancement.py`,
which uses the existing `black_tophat.py` and `unsharp_mask.py` modules. Keep
these files together in this repository; `enhance.py` alone is not standalone.
The required packages (`h5py`, `numpy`, and `scikit-image`) are already declared
in `pyproject.toml`.

### Install and run

From the repository root, install the project with uv:

```bash
uv sync
uv run python scripts/enhance.py --help
```

Pass your own input and output paths. Relative paths are relative to your current
working directory; absolute paths also work. No personal root path or `.env` /
`DATA_DIR` setting is required for this script. These examples assume your files
are in `data/nisar_data/` (replace `input.h5` with your actual filename):

```bash
# Emphasize dark features using black top-hat (the default).
uv run python scripts/enhance.py \
  data/nisar_data/input.h5 data/output/input_black_tophat.h5

# Sharpen details with unsharp masking.
uv run python scripts/enhance.py \
  data/nisar_data/input.h5 data/output/input_unsharp.h5 --method unsharp

# Reduce working memory by using smaller chunks.
uv run python scripts/enhance.py \
  data/nisar_data/input.h5 data/output/input_small_chunks.h5 --chunk-size 512
```

`--chunk-size` defaults to 2048 pixels per side and must be a positive multiple
of 512. Smaller chunks generally use less memory but require more processing.
The script reads the input twice: first to estimate a global median, then to
filter each chunk. It prints progress every ten chunks and at completion.
Allow enough disk space for another full-frame HDF5 file; compressed size varies.

### Supported input and processing

The input must contain the 2D linear-power `HHHH` dataset under
`science/LSAR/GCOV/grids/frequencyA`, plus `xCoordinates`, `yCoordinates`,
`xCoordinateSpacing`, `yCoordinateSpacing`, and `projection` in that group.
Other bands, frequency B, and other NISAR product types are not supported by
this script. If `inputDataExceptionMask` exists, only pixels whose mask equals
zero are used. Nonfinite and nonpositive values are also treated as invalid.

Valid power values are converted with `10 * log10(power)`. Invalid pixels are
filled temporarily with the frame's approximate median (0.01 dB histogram bins),
then restored to NaN in the output. Filtering uses radius 4, unsharp amount 1.0,
and a 32-pixel overlap around chunks to avoid artificial boundaries. Black
top-hat subtracts the filter response from the dB image to emphasize dark
features. These settings preserve the original local script's processing.

### Output and interrupted runs

The output is a **derived enhanced HDF5 image**, not a PNG, GeoTIFF, or complete
NISAR GCOV product. It contains the enhanced `HHHH` band at the original dataset
path, the coordinates/projection, and processing attributes. Other source bands,
the exception mask, and full product metadata are not copied. Values are stored
as float32 linear encoding (`10 ** (enhanced_dB / 10)`), with invalid pixels set
to NaN. They are enhanced display/analysis values, **not calibrated original
backscatter**. Use the original product for measurements requiring calibration.

Output attributes record the source path, method, filter settings, valid-pixel
count, and completion status. A `.partial` file is written beside the requested
output and renamed only after processing succeeds. Existing output and partial
files are refused. If interrupted, inspect or move the partial file before
retrying; resuming is not supported. An error is raised for missing datasets,
no valid pixels, values outside the histogram range (-500 to 500 dB), or enhanced
values that cannot be represented as positive float32 power.

Upload the script, its new shared module, and documentation to GitHub together.
Keep input data, generated HDF5 outputs, and partial files out of the commit.
