#!/usr/bin/env python3
"""Per-threat-category ASR analysis from per-task data.

Reads: data/per_task_first_run.json, data/desktop_tasks.csv, data/mobile_tasks.csv
Outputs: Per-threat ASR table for all models.

Breaks ASR (paper §3.4) down over the 10 safety threat categories of the risk
framework (paper §3.1, enumerated in Appendix A); analysis in §4.4.

Usage:
    python per_threat_10run.py
"""

import json
from collections import defaultdict

from common import DATA_DIR, load_threat_map, MODELS_ORDER, SHORT_NAMES


def main():
    threat_map = load_threat_map()

    with open(DATA_DIR / "per_task_first_run.json") as f:
        raw = json.load(f)

    per_task = {}
    for key, scores in raw.items():
        parts = key.split("|")
        per_task[(parts[0], parts[1], parts[2])] = scores

    threat_results = defaultdict(lambda: defaultdict(lambda: {"hit": 0, "total": 0}))

    for model in MODELS_ORDER:
        for platform in ["desktop", "mobile"]:
            mal = per_task.get((model, platform, "malicious"), {})
            for tid, score in mal.items():
                threat = threat_map.get((platform, tid))
                if threat:
                    threat_results[threat][model]["total"] += 1
                    threat_results[threat][model]["hit"] += score

    frontier = MODELS_ORDER[:4]
    threat_order = sorted(
        threat_results.keys(),
        key=lambda t: sum(
            threat_results[t][m]["hit"] / max(threat_results[t][m]["total"], 1)
            for m in frontier
        )
        / len(frontier),
        reverse=True,
    )

    header = [SHORT_NAMES.get(m, m[:15]) for m in frontier]
    print(f"{'Threat':<25} " + " ".join(f"{h:<15}" for h in header) + " N")
    print("-" * 90)

    for threat in threat_order:
        vals = []
        for mid in frontier:
            d = threat_results[threat][mid]
            if d["total"] > 0:
                vals.append(f"{d['hit'] / d['total'] * 100:.1f}%")
            else:
                vals.append("--")
        n = max(threat_results[threat][m]["total"] for m in frontier)
        print(f"  {threat:<25} " + " ".join(f"{v:<15}" for v in vals) + f" {n}")


if __name__ == "__main__":
    main()
