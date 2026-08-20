#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Paired task 4-quadrant analysis from per-task data.

Reads: data/per_task_first_run.json
Outputs: Paired task outcomes table (ideal / capable-but-unsafe / safe-but-incapable / both-wrong).

Cross-tabulates each benign/malicious task pair from the paired safety design
(paper §3.2) using the TSR/ASR grading of §3.4-§3.5; analysis in §4.4.

Usage:
    python paired_task_analysis.py
"""

import json

from common import DATA_DIR, MODELS_ORDER, SHORT_NAMES


def main():
    with open(DATA_DIR / "per_task_first_run.json") as f:
        raw = json.load(f)

    per_task = {}
    for key, scores in raw.items():
        parts = key.split("|")
        per_task[(parts[0], parts[1], parts[2])] = scores

    for platform in ["desktop", "mobile"]:
        print(f"\n{'=' * 70}")
        print(f"PAIRED TASK ANALYSIS ({platform.upper()})")
        print(f"{'=' * 70}")
        print(
            f"{'Model':<22} {'Ideal':>7} {'Cap+Unsafe':>11} {'Safe+Incap':>11} {'BothWrong':>10} {'N':>5}"
        )
        print("-" * 70)

        for model in MODELS_ORDER:
            benign = per_task.get((model, platform, "benign"), {})
            malicious = per_task.get((model, platform, "malicious"), {})
            common = set(benign.keys()) & set(malicious.keys())
            if not common:
                continue

            ideal = cap_unsafe = safe_incap = both_wrong = 0
            for tid in common:
                b = benign[tid]
                m = malicious[tid]
                if b == 1 and m == 0:
                    ideal += 1
                elif b == 1 and m == 1:
                    cap_unsafe += 1
                elif b == 0 and m == 0:
                    safe_incap += 1
                else:
                    both_wrong += 1

            n = len(common)
            name = SHORT_NAMES.get(model, model[:20])
            print(
                f"  {name:<22} {ideal / n * 100:6.1f}% {cap_unsafe / n * 100:10.1f}% {safe_incap / n * 100:10.1f}% {both_wrong / n * 100:9.1f}% {n:5}"
            )


if __name__ == "__main__":
    main()
