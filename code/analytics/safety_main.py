#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Compute main results table from 10-run aggregate CSV.

Reads: data/results_all.csv
Outputs: Main results table (mean +/- 95% CI over 10 runs) for paper Table 1.

Metrics — TSR, ASR, FRR, and the ADEPTS-BENCH Score — are defined in paper §3.4
and produced by the grading methodology of §3.5. The score formula is
common.compute_score (harmonic mean of TSR and 1-ASR, paper §3.4).

Usage:
    python main_results_10run.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def main():
    rows = load_results()
    base = [r for r in rows if r["has_unsafe_tool"]]

    grouped = defaultdict(list)
    for r in base:
        key = (r["model_ids"], r["platform"], r["task_type"])
        grouped[key].append(r)

    for platform in ["desktop", "mobile"]:
        n_tasks = EXPECTED_N[platform]
        print(f"\n{'=' * 80}")
        print(f"{platform.upper()} ({n_tasks} tasks per condition)")
        print(f"{'=' * 80}")
        print(
            f"{'Model':<22} {'TSR%':>6} {'±CI':>5} {'ASR%':>6} {'±CI':>5} {'Score%':>7} {'FRR%':>6} {'Unsafe%(mal)':>12} {'Runs':>5}"
        )
        print("-" * 80)

        for model in MODELS_ORDER:
            b_key = (model, platform, "benign")
            m_key = (model, platform, "malicious")
            if b_key not in grouped or m_key not in grouped:
                continue

            tsr_runs = [float(r["score"]) for r in grouped[b_key]]
            asr_runs = [float(r["score"]) for r in grouped[m_key]]
            frr_runs = [
                float(r["act_StatusAction(unsafe)"]) * 100 for r in grouped[b_key]
            ]
            unsafe_mal = [
                float(r["act_StatusAction(unsafe)"]) * 100 for r in grouped[m_key]
            ]

            tsr = statistics.mean(tsr_runs) * 100
            asr = statistics.mean(asr_runs) * 100
            tsr_c = ci95(tsr_runs) * 100
            asr_c = ci95(asr_runs) * 100
            score = compute_score(tsr / 100, asr / 100) * 100
            frr = statistics.mean(frr_runs)
            unsafe = statistics.mean(unsafe_mal)

            name = SHORT_NAMES.get(model, model[:20])
            print(
                f"  {name:<22} {tsr:5.1f} {tsr_c:5.1f} {asr:5.1f} {asr_c:5.1f} {score:6.1f} {frr:6.1f} {unsafe:11.1f} {len(tsr_runs):5}"
            )


if __name__ == "__main__":
    main()
