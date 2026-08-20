#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Convert benchmark results to the format expected by analysis scripts.

Reads from results/ and data/ directories, writes to data/analytics/.

Safety: Aggregates per-model/per-dataset results.json into results_all.csv
        and per_task_first_run.json.
Disambiguation: Converts scored_generations.json into disambig_{platform}_final.json,
                and copies ground truth into tasks_{platform}_gt.json.

Usage:
    python code/analytics/convert_results.py
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import REPO_ROOT, log as _log, reset_log
from common import canonicalize_model_id

SAFETY_RESULTS = os.path.join(REPO_ROOT, "results", "safety")
DISAMBIG_RESULTS = os.path.join(REPO_ROOT, "results", "disambiguation")
SAFETY_DATA = os.path.join(REPO_ROOT, "data", "safety")
DISAMBIG_DATA = os.path.join(REPO_ROOT, "data", "disambiguation")
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "analytics")


def convert_safety():
    """Convert safety results to results_all.csv and per_task_first_run.json."""
    if not os.path.isdir(SAFETY_RESULTS):
        _log("No safety results found, skipping")
        return

    csv_rows = []
    per_task = {}

    for dataset_name in sorted(os.listdir(SAFETY_RESULTS)):
        dataset_dir = os.path.join(SAFETY_RESULTS, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue

        for model_name in sorted(os.listdir(dataset_dir)):
            results_file = os.path.join(dataset_dir, model_name, "results.json")
            if not os.path.exists(results_file):
                continue

            with open(results_file) as f:
                data = json.load(f)

            results = data.get("results", [])
            metadata = data.get("metadata", {})
            task_name = metadata.get("input_file", dataset_name)
            model_id = canonicalize_model_id(metadata.get("model", model_name))

            if not results:
                continue

            attempted = [r for r in results if r.get("error") is None and r.get("category") != "api_error"]
            n_attempted = len(attempted)
            if n_attempted == 0:
                continue

            correct = sum(1 for r in attempted if r.get("reward", 0.0) == 1.0)
            score = correct / n_attempted

            unsafe_count = sum(
                1 for r in attempted
                if "StatusAction" in r.get("parsed_action", "") and "unsafe" in r.get("parsed_action", "")
            )
            act_unsafe = unsafe_count / n_attempted if n_attempted else 0

            csv_rows.append({
                "task_name": task_name.replace(".jsonl", ""),
                "model_ids": model_id,
                "n": n_attempted,
                "score": f"{score:.4f}",
                "act_StatusAction(unsafe)": f"{act_unsafe:.4f}",
            })

            # Per-task data (first run)
            is_benign = "benign" in task_name
            platform = "mobile" if "mobile" in task_name else "desktop"
            task_type = "benign" if is_benign else "malicious"
            pt_key = f"{model_id}|{platform}|{task_type}"

            if pt_key not in per_task:
                per_task[pt_key] = {}
            for r in attempted:
                sid = str(r.get("sample_id", 0))
                per_task[pt_key][sid] = r.get("reward", 0.0)

    if csv_rows:
        csv_path = os.path.join(OUTPUT_DIR, "results_all.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["task_name", "model_ids", "n", "score", "act_StatusAction(unsafe)"])
            writer.writeheader()
            writer.writerows(csv_rows)
        _log(f"  Wrote {len(csv_rows)} rows to {csv_path}")

    if per_task:
        pt_path = os.path.join(OUTPUT_DIR, "per_task_first_run.json")
        with open(pt_path, "w") as f:
            json.dump(per_task, f, indent=2)
        _log(f"  Wrote {len(per_task)} keys to {pt_path}")


def convert_safety_tasks():
    """Copy task CSVs for threat category mapping."""
    for platform, src_name, dst_name in [
        ("desktop", "tasks_desktop.json", "desktop_tasks.csv"),
        ("mobile", "tasks_mobile.json", "mobile_tasks.csv"),
    ]:
        src = os.path.join(SAFETY_DATA, src_name)
        if not os.path.exists(src):
            continue
        with open(src) as f:
            tasks = json.load(f)

        dst = os.path.join(OUTPUT_DIR, dst_name)
        if not tasks:
            continue

        fieldnames = ["task_id"]
        for key in ["threat_category", "risk_severity_level", "app", "user_goal", "malicious_intent", "victim", "action_object"]:
            if key in tasks[0]:
                fieldnames.append(key)

        with open(dst, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for t in tasks:
                writer.writerow(t)
        _log(f"  Wrote {len(tasks)} tasks to {dst}")


def convert_disambiguation():
    """Convert disambiguation results to the format expected by analytics scripts."""
    if not os.path.isdir(DISAMBIG_RESULTS):
        _log("No disambiguation results found, skipping")
        return

    for platform in ["mobile", "desktop"]:
        scored_file = os.path.join(DISAMBIG_RESULTS, f"{platform}_scored_generations.json")
        if not os.path.exists(scored_file):
            continue

        with open(scored_file) as f:
            data = json.load(f)

        dst = os.path.join(OUTPUT_DIR, f"disambig_{platform}_final.json")
        with open(dst, "w") as f:
            json.dump(data, f, indent=2)
        _log(f"  Wrote {len(data)} entries to {dst}")

    # Copy ground truth
    for platform, src_name, dst_name in [
        ("mobile", "tasks_mobile.json", "tasks_mobile_gt.json"),
        ("desktop", "tasks_desktop.json", "tasks_desktop_gt.json"),
    ]:
        src = os.path.join(DISAMBIG_DATA, src_name)
        if not os.path.exists(src):
            continue
        dst = os.path.join(OUTPUT_DIR, dst_name)
        with open(src) as f:
            data = json.load(f)
        with open(dst, "w") as f:
            json.dump(data, f, indent=2)
        _log(f"  Copied {len(data)} tasks to {dst}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    reset_log()

    _log("Converting safety results", step=True)
    convert_safety()
    convert_safety_tasks()

    _log("Converting disambiguation results", step=True)
    convert_disambiguation()

    _log("Done")


if __name__ == "__main__":
    main()
