# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Reset ADEPTS codebase by removing generated data, pre-processing, and results.

Usage:
    python code/reset.py              # Reset everything
    python code/reset.py data         # Reset downloaded data only
    python code/reset.py preprocess   # Reset pre-processed images and datasets only
    python code/reset.py results      # Reset benchmark results only
    python code/reset.py all          # Reset everything (same as no args)
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_DIR, RESULTS_DIR, SAFETY_DATA_DIR, log as _log, reset_log

TARGETS = {
    "data": [
        (DATA_DIR, "dir"),
    ],
    "preprocess": [
        (os.path.join(SAFETY_DATA_DIR, "pre_processed_images"), "dir"),
        (os.path.join(SAFETY_DATA_DIR, "dataset"), "dir"),
    ],
    "results": [
        (RESULTS_DIR, "dir"),
    ],
}


def reset(categories: list[str]) -> None:
    removed = 0
    skipped = 0
    for category in categories:
        targets = TARGETS.get(category, [])
        _log(f"Resetting {category}...", step=True)
        for path, kind in targets:
            if not os.path.exists(path):
                skipped += 1
                continue
            if kind == "dir":
                shutil.rmtree(path)
                _log(f"  Removed directory: {path}")
            else:
                os.remove(path)
                _log(f"  Removed file: {path}")
            removed += 1

    _log(f"Done: {removed} removed, {skipped} already clean")


def main() -> None:
    valid = list(TARGETS.keys()) + ["all"]
    args = sys.argv[1:]

    if not args or "all" in args:
        categories = list(TARGETS.keys())
    else:
        for arg in args:
            if arg not in valid:
                raise SystemExit(f"Error: unknown target '{arg}'. Valid targets: {', '.join(valid)}")
        categories = args

    reset_log()
    _log(f"Resetting: {', '.join(categories)}")
    print("This will delete generated files. Continue? [y/N] ", end="")
    if input().strip().lower() != "y":
        _log("Aborted.")
        return

    reset(categories)


if __name__ == "__main__":
    main()
