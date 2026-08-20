#!/usr/bin/env python3
"""Run ADEPTS analysis scripts.

Usage:
    python code/analytics/analyze.py                # Run all analyses
    python code/analytics/analyze.py safety          # Safety analyses only
    python code/analytics/analyze.py disambiguation  # Disambiguation analyses only
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import log as _log, reset_log

from convert_results import main as convert_main
from safety_main import main as safety_main_fn
from safety_ablation import main as safety_ablation_fn
from safety_passk import main as safety_passk_fn
from safety_paired import main as safety_paired_fn
from safety_per_threat import main as safety_per_threat_fn
from safety_failure_spectrum import main as safety_failure_spectrum_fn
from generate_pareto import main as generate_pareto_fn
from disambig_main import main as disambig_main_fn
from disambig_analysis import main as disambig_analysis_fn

SAFETY_SCRIPTS = [
    ("Table 1: TSR, ASR, ADEPTS Score", safety_main_fn),
    ("Table 2: Tool ablation", safety_ablation_fn),
    ("Pass@k worst-case", safety_passk_fn),
    ("Table 6: Paired 2x2", safety_paired_fn),
    ("Table 7: Per-threat ASR", safety_per_threat_fn),
    ("Figure 3: Failure hierarchy", safety_failure_spectrum_fn),
    ("Figure 4: Pareto frontier", generate_pareto_fn),
]

DISAMBIG_SCRIPTS = [
    ("Table 3: F1 and Delta", disambig_main_fn),
    ("Failure mode analysis", disambig_analysis_fn),
]


def run_scripts(scripts):
    for label, fn in scripts:
        _log(f"\n{'=' * 70}")
        _log(f"  {label}")
        _log(f"{'=' * 70}")
        try:
            fn()
        except Exception as e:
            _log(f"  Skipped: {type(e).__name__}: {e}")


def main():
    valid = ["safety", "disambiguation", "all"]
    args = sys.argv[1:]
    targets = args if args else ["all"]

    for t in targets:
        if t not in valid:
            raise SystemExit(f"Error: unknown target '{t}'. Valid targets: {', '.join(valid)}")

    reset_log()

    _log("Converting benchmark results to analytics format", step=True)
    convert_main()

    if "safety" in targets or "all" in targets:
        _log("Running safety analyses", step=True)
        run_scripts(SAFETY_SCRIPTS)

    if "disambiguation" in targets or "all" in targets:
        _log("Running disambiguation analyses", step=True)
        run_scripts(DISAMBIG_SCRIPTS)

    _log("Done")


if __name__ == "__main__":
    main()
