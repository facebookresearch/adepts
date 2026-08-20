"""Print safety benchmark results from the results directory.

Usage:
    python code/safety/print_results.py
    python code/safety/print_results.py --results-dir results/safety
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import SAFETY_RESULTS_DIR, SummaryPrinter, compute_summary_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Print safety benchmark results")
    parser.add_argument("--results-dir", default=SAFETY_RESULTS_DIR, help=f"Results directory (default: {SAFETY_RESULTS_DIR})")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Error: results directory not found: {args.results_dir}")
        raise SystemExit(1)

    printer = SummaryPrinter()

    for dataset_name in sorted(os.listdir(args.results_dir)):
        dataset_dir = os.path.join(args.results_dir, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        for model_name in sorted(os.listdir(dataset_dir)):
            model_dir = os.path.join(dataset_dir, model_name)
            if not os.path.isdir(model_dir):
                continue
            results_file = os.path.join(model_dir, "results.json")
            if not os.path.exists(results_file):
                continue
            with open(results_file) as f:
                data = json.load(f)
            sample_results = data.get("results", [])
            input_file = data.get("metadata", {}).get("input_file", dataset_name)
            row = compute_summary_row(input_file, model_name, sample_results)
            if row:
                printer.add_row(row)

    printer.print_table()


if __name__ == "__main__":
    main()
