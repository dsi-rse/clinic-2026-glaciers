"""Enhance a complete NISAR GCOV frequency A HHHH frame.

Run from the repository root with:
    uv run python scripts/enhance.py INPUT.h5 OUTPUT.h5 --method black-tophat

See scripts/README.md for dependencies, output details, and limitations.
"""

import argparse
from pathlib import Path

from utils.inference_strategies.full_frame_enhancement import enhance_frame


def main() -> None:
    """Parse file paths and enhancement options, then process one frame."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input NISAR GCOV HDF5 file")
    parser.add_argument("output", type=Path, help="New derived HDF5 output file")
    parser.add_argument(
        "--method",
        choices=["black-tophat", "unsharp"],
        default="black-tophat",
        help="Enhancement filter (default: black-tophat)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2048,
        help="Pixels per chunk side; positive multiple of 512 (default: 2048)",
    )
    args = parser.parse_args()
    try:
        enhance_frame(args.input, args.output, args.method, args.chunk_size)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Enhancement failed: {error}\n")


if __name__ == "__main__":
    main()
