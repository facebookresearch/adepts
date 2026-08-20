#!/usr/bin/env python3
"""Pass@k worst-case safety analysis from per-task data.

Reads: data/pass_k_data.json (per-task scores across 10 runs)
Outputs: Worst-case ASR table, flaky task counts, compliance frequency distribution.

Worst-case (pass@k) view of ASR (paper §3.4) across the 10 runs per condition
described in §4.1; additional safety analysis in §4.4.

Usage:
    python pass_at_k.py
"""

import json
from collections import defaultdict

from common import DATA_DIR, MODELS_ORDER, SHORT_NAMES


def main():
    with open(DATA_DIR / "pass_k_data.json") as f:
        raw = json.load(f)

    print(
        f"{'Model':<22} {'Plat':<8} {'ASR@1':>7} {'Worst':>7} {'Best':>7} {'Gap':>7} {'Flaky':>6} {'Always':>7} {'Never':>7}"
    )
    print("-" * 84)

    for model in MODELS_ORDER:
        for platform in ["desktop", "mobile"]:
            key = f"{model}|{platform}"
            if key not in raw:
                continue
            ts = raw[key]
            total = len(ts)
            any_comply = sum(1 for s in ts.values() if any(x > 0 for x in s))
            all_comply = sum(1 for s in ts.values() if all(x > 0 for x in s))
            never = total - any_comply
            flaky = any_comply - all_comply
            mean_asr = sum(sum(s) / len(s) for s in ts.values()) / total * 100
            worst = any_comply / total * 100
            best = all_comply / total * 100

            name = SHORT_NAMES.get(model, model[:20])
            print(
                f"  {name:<22} {platform:<8} {mean_asr:6.1f}% {worst:6.1f}% {best:6.1f}% {worst - mean_asr:+6.1f} {flaky:6} {all_comply:7} {never:7}"
            )

    # Compliance frequency distribution for selected models
    print("\n\nCompliance Frequency Distribution (desktop, how many of k runs comply):")
    for model in [
        "gemini-3.1-pro",
        "claude-4.7-opus",
        "gpt-5.4",
        "gemini-2.5-cu",
    ]:
        key = f"{model}|desktop"
        if key not in raw:
            continue
        ts = raw[key]
        total = len(ts)
        k = len(list(ts.values())[0])

        freq = defaultdict(int)
        for scores in ts.values():
            n_comply = sum(1 for s in scores if s > 0)
            freq[n_comply] += 1

        name = SHORT_NAMES.get(model, model[:20])
        print(f"\n  {name} (k={k}):")
        for i in range(k + 1):
            count = freq.get(i, 0)
            bar = "#" * (count // 5)
            print(f"    {i:2d}/{k}: {count:4d} ({count / total * 100:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
