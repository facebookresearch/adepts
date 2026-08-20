#!/usr/bin/env python3
"""Compute disambiguation results table (F1 and Delta) for both platforms, both modes.

Reads:
    data/disambig_mobile_final.json   - Mobile task-level results (Mode 1 + Mode 2)
    data/disambig_desktop_final.json  - Desktop task-level results
    data/tasks_mobile_gt.json         - Mobile ground truth (381 tasks)
    data/tasks_desktop_gt.json        - Desktop ground truth (363 tasks)

Outputs: Disambiguation results table (paper Table 3) with F1 and Delta.

Metrics are defined in paper §3.4 ("Disambiguation metrics"); computed by
common.compute_disambig_prf1_delta.
    F1: Harmonic mean of precision and recall (paper §3.4).
        Precision = (generated questions matching a GT item) / (total generated questions)
        Recall = (unique GT items matched by generated questions) / (total GT items)
        Matching is determined by an LLM judge assessing semantic equivalence
        (Gemini 3.1 Pro in the paper, §3.5; configurable via --check-questions-model).

    Delta (Severity Calibration Error): per-component absolute error on matched
        items, paper Eq. (1) in §3.4.
        Delta = mean(|gen_obv - gt_obv| + |gen_con - gt_con|)
        Computed over all matched (generated, GT) question pairs.
        Per-component formulation prevents cancellation between overestimated
        obviousness and underestimated consequence.

Usage:
    python disambig_main.py
"""

import json
from collections import defaultdict



import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_DIR, DISAMBIG_MODEL_MAP, compute_disambig_prf1_delta, parse_disambig_key as parse_key


def compute_metrics(data, gt_dict, ptype_filter):
    """Compute F1 and Delta for one (dataset, prompt_type) combination."""
    mode_data = defaultdict(dict)
    for k, v in data.items():
        model_raw, ptype, tid = parse_key(k)
        if model_raw not in DISAMBIG_MODEL_MAP:
            continue
        if ptype != ptype_filter:
            continue
        mode_data[DISAMBIG_MODEL_MAP[model_raw]][tid] = v

    results = {}
    for model in sorted(mode_data.keys()):
        entries = mode_data[model]
        n_tasks = len(entries)
        n_clarified = sum(
            1
            for v in entries.values()
            if v.get("gen_clarifications") and len(v["gen_clarifications"]) > 0
        )

        pairs = []
        for tid, v in entries.items():
            gt_task = gt_dict.get(tid)
            if not gt_task:
                continue
            pairs.append((v.get("gen_clarifications", []), gt_task.get("clarifications", [])))

        m = compute_disambig_prf1_delta(pairs)
        results[model] = {
            "n": n_tasks,
            "cr": n_clarified / n_tasks if n_tasks else 0,
            **m,
        }
    return results


def main():
    datasets = []
    for name, data_file, gt_file in [
        ("Mobile", "disambig_mobile_final.json", "tasks_mobile_gt.json"),
        ("Desktop", "disambig_desktop_final.json", "tasks_desktop_gt.json"),
    ]:
        data_path = DATA_DIR / data_file
        gt_path = DATA_DIR / gt_file
        if not data_path.exists() or not gt_path.exists():
            print(f"Skipping {name}: {data_file} or {gt_file} not found in data/")
            continue
        with open(data_path) as f:
            data = json.load(f)
        with open(gt_path) as f:
            gt = {t["id"]: t for t in json.load(f)}
        datasets.append((name, data, gt))

    for name, data, gt in datasets:
        for mode_label, ptypes in [
            ("Mode 2 (with severity scoring)", ["WITHOUT_COT_SCORE", "WITH_SCORE"]),
            ("Mode 1 (questions only)", ["WITHOUT_COT_NO_SCORE", "NO_SCORE"]),
        ]:
            results = {}
            for ptype in ptypes:
                results = compute_metrics(data, gt, ptype)
                if results:
                    break
            if not results:
                continue

            print(f"\n{'='*70}")
            print(f"  {name} - {mode_label}")
            print(f"{'='*70}")
            print(
                f"{'Model':<15} {'n':>4} {'CR':>7} {'Prec':>7} {'Recall':>7} "
                f"{'F1':>7} {'Delta':>7} {'n_d':>5}"
            )

            for m in [
                "Gemini 3.1",
                "Claude 4.7",
                "Gemini CU",
                "GPT-5.4",
                "Qwen 235B",
                "Qwen 4B",
                "Qwen 8B",
            ]:
                if m not in results:
                    continue
                r = results[m]
                d_str = f"{r['delta']:7.2f}" if r["n_delta"] > 0 else "    N/A"
                print(
                    f"{m:<15} {r['n']:>4} {r['cr']:>6.1%} {r['precision']:>7.3f} "
                    f"{r['recall']:>7.3f} {r['f1']:>7.3f} {d_str} {r['n_delta']:>5}"
                )


if __name__ == "__main__":
    main()
