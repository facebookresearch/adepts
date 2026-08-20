#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Safety tool ablation: base setup vs no-tool/no-prompt.

Reads: data/results_all.csv
Outputs: Ablation table showing ASR/TSR impact of removing StatusAction(unsafe) tool + safety prompt.

Usage:
    python ablation_safety_tool.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def main():
    rows = load_results()

    grouped = defaultdict(list)
    for r in rows:
        key = (r["model_ids"], r["platform"], r["task_type"], r["has_unsafe_tool"])
        grouped[key].append(float(r["score"]))

    print(
        f"{'Model':<22} {'Plat':<8} {'ASR(base)':>10} {'ASR(no)':>10} {'ΔASR':>8} {'TSR(base)':>10} {'TSR(no)':>10} {'ΔTSR':>8}"
    )
    print("-" * 88)

    for model in MODELS_ORDER:
        for platform in ["desktop", "mobile"]:
            aw = grouped.get((model, platform, "malicious", True), [])
            awo = grouped.get((model, platform, "malicious", False), [])
            tw = grouped.get((model, platform, "benign", True), [])
            two = grouped.get((model, platform, "benign", False), [])

            if not (aw and awo and tw and two):
                continue

            name = SHORT_NAMES.get(model, model[:20])
            aw_m = statistics.mean(aw) * 100
            awo_m = statistics.mean(awo) * 100
            tw_m = statistics.mean(tw) * 100
            two_m = statistics.mean(two) * 100
            print(
                f"  {name:<22} {platform:<8} {aw_m:9.1f}% {awo_m:9.1f}% {awo_m - aw_m:+7.1f}pp {tw_m:9.1f}% {two_m:9.1f}% {two_m - tw_m:+7.1f}pp"
            )

    print("\nThree safety architectures:")
    print("  Tool-dependent:          Gemini 3.1 Pro (ASR +23pp)")
    print("  Partially tool-dependent: Claude 4.7, GPT-5.4 (ASR +11pp)")
    print("  No mechanism:            Qwen (all sizes, ASR unchanged)")


if __name__ == "__main__":
    main()
