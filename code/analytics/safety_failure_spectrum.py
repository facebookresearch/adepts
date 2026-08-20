#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Deep failure mode analysis: content-level patterns in safety failures.

Reads: data/per_task_first_run.json, data/desktop_tasks.csv, data/mobile_tasks.csv
Outputs: Failure spectrum analysis, all-comply/all-refuse categorization,
         model-specific vulnerability profiles, IRT misfits, cross-platform consistency.

Corresponds to the Safety Failure Spectrum (paper §4.6, Figure 3).

Usage:
    python failure_mode_analysis.py
"""

import json
from collections import Counter, defaultdict

from common import DATA_DIR, MODELS_ORDER, SHORT_NAMES


def load_per_task():
    with open(DATA_DIR / "per_task_first_run.json") as f:
        raw = json.load(f)
    per_task = {}
    for key, scores in raw.items():
        parts = key.split("|")
        per_task[(parts[0], parts[1], parts[2])] = scores
    return per_task


def load_task_metadata():
    import csv

    task_meta = {}
    for platform, fname in [
        ("desktop", "desktop_tasks.csv"),
        ("mobile", "mobile_tasks.csv"),
    ]:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for r in csv.DictReader(f):
                tid = r.get("task_id", "").strip()
                if not tid:
                    continue
                task_meta[(platform, tid)] = {
                    "threat": r.get("threat_category", "")
                    .strip()
                    .title()
                    .replace("Dangerous Command Lines", "Dangerous Commands")
                    .replace("Dangerous Command Line", "Dangerous Commands"),
                    "app": r.get("app", "").strip(),
                    "user_goal": r.get("user_goal", "").strip(),
                    "malicious_intent": r.get("malicious_intent", "").strip(),
                    "victim": r.get("victim", "").strip(),
                    "action_object": r.get("action_object", "").strip(),
                    "risk_severity": r.get("risk_severity_level", "").strip(),
                }
    return task_meta


FRONTIER = MODELS_ORDER[:4]


def analyze_all_comply(per_task, task_meta):
    """Tasks where ALL models comply with unsafe instruction."""
    print("\n" + "=" * 80)
    print("ALL-COMPLY TASKS (no model detects threat)")
    print("=" * 80)

    for platform in ["desktop", "mobile"]:
        task_scores = defaultdict(dict)
        for model in MODELS_ORDER:
            mal = per_task.get((model, platform, "malicious"), {})
            for tid, score in mal.items():
                task_scores[tid][model] = score

        all_comply = [
            (tid, task_meta.get((platform, tid), {}))
            for tid, scores in task_scores.items()
            if len(scores) >= 6 and all(s == 1 for s in scores.values())
        ]

        print(f"\n--- {platform.upper()}: {len(all_comply)} all-comply tasks ---")

        threats = Counter(m.get("threat", "?") for _, m in all_comply)
        print("  By threat:")
        for t, c in threats.most_common(10):
            print(f"    {t}: {c}")

        apps = Counter(m.get("app", "?") for _, m in all_comply)
        print("  By app:")
        for a, c in apps.most_common(10):
            print(f"    {a}: {c}")

        # Categorize by intent pattern
        categories = defaultdict(list)
        for tid, task_info in all_comply:
            intent = task_info.get("malicious_intent", "").lower()
            if any(
                w in intent
                for w in ["mass", "overload", "duplicate", "all ", "bulk", "every"]
            ):
                categories["Excessive/mass action"].append((tid, task_info))
            elif any(
                w in intent
                for w in [
                    "wrong",
                    "incorrect",
                    "different",
                    "instead",
                    "default",
                    "doesn't pay",
                ]
            ):
                categories["Subtle wrong target"].append((tid, task_info))
            elif any(
                w in intent
                for w in ["forward", "send", "share", "leak", "expose", "personal"]
            ):
                categories["Data exposure"].append((tid, task_info))
            elif any(
                w in intent
                for w in [
                    "disable",
                    "deactivat",
                    "reduce",
                    "lower",
                    "factory",
                    "reset",
                ]
            ):
                categories["Security downgrade"].append((tid, task_info))
            elif any(
                w in intent
                for w in ["trial", "in-app", "subscription", "charges", "cost"]
            ):
                categories["Hidden cost"].append((tid, task_info))
            elif any(
                w in intent for w in ["phish", "fake", "disguised", "trick", "scam"]
            ):
                categories["Deceptive content"].append((tid, task_info))
            else:
                categories["Other"].append((tid, task_info))

        print("  By intent pattern:")
        for cat, tasks in sorted(categories.items(), key=lambda x: -len(x[1])):
            print(f"    {cat}: {len(tasks)}")


def analyze_all_refuse(per_task, task_meta):
    """Tasks where ALL models refuse (universally detected)."""
    print("\n" + "=" * 80)
    print("ALL-REFUSE TASKS (universally detected)")
    print("=" * 80)

    for platform in ["desktop", "mobile"]:
        task_scores = defaultdict(dict)
        for model in MODELS_ORDER:
            mal = per_task.get((model, platform, "malicious"), {})
            for tid, score in mal.items():
                task_scores[tid][model] = score

        all_refuse = [
            (tid, task_meta.get((platform, tid), {}))
            for tid, scores in task_scores.items()
            if len(scores) >= 6 and all(s == 0 for s in scores.values())
        ]

        print(f"\n--- {platform.upper()}: {len(all_refuse)} all-refuse tasks ---")
        threats = Counter(m.get("threat", "?") for _, m in all_refuse)
        print("  By threat:")
        for t, c in threats.most_common(10):
            print(f"    {t}: {c}")


def analyze_model_unique_vulnerabilities(per_task, task_meta):
    """Tasks where only one frontier model complies (others refuse)."""
    print("\n" + "=" * 80)
    print("MODEL-SPECIFIC UNIQUE VULNERABILITIES")
    print("=" * 80)

    for platform in ["desktop", "mobile"]:
        print(f"\n--- {platform.upper()} ---")
        task_scores = defaultdict(dict)
        for model in FRONTIER:
            mal = per_task.get((model, platform, "malicious"), {})
            for tid, score in mal.items():
                task_scores[tid][model] = score

        for target in FRONTIER:
            unique = []
            for tid, scores in task_scores.items():
                if len(scores) < 4:
                    continue
                others = [s for m, s in scores.items() if m != target]
                if scores.get(target, 0) == 1 and all(s == 0 for s in others):
                    unique.append((tid, task_meta.get((platform, tid), {})))

            name = SHORT_NAMES.get(target, target[:20])
            print(f"\n  {name} uniquely complies: {len(unique)}")
            if unique:
                threats = Counter(m.get("threat", "?") for _, m in unique)
                print(f"    Top threats: {dict(threats.most_common(5))}")
                for tid, task_info in unique[:3]:
                    print(
                        f"    task_{tid}: {task_info.get('app', '?')} | {task_info.get('user_goal', '')[:60]}"
                    )


def analyze_irt_misfits(per_task, task_meta):
    """Tasks where open-source passes but frontier fails (over-refusal)."""
    print("\n" + "=" * 80)
    print("IRT MISFITS (frontier over-refuses, open-source completes)")
    print("=" * 80)

    oss_models = MODELS_ORDER[4:]
    misfits = []

    for platform in ["desktop", "mobile"]:
        for tid in set().union(
            *(per_task.get((m, platform, "benign"), {}).keys() for m in MODELS_ORDER)
        ):
            f_scores = [
                per_task.get((m, platform, "benign"), {}).get(tid) for m in FRONTIER
            ]
            q_scores = [
                per_task.get((m, platform, "benign"), {}).get(tid) for m in oss_models
            ]
            f_scores = [s for s in f_scores if s is not None]
            q_scores = [s for s in q_scores if s is not None]
            if f_scores and q_scores:
                if (
                    sum(f_scores) / len(f_scores) < 0.5
                    and sum(q_scores) / len(q_scores) > 0.5
                ):
                    task_info = task_meta.get((platform, tid), {})
                    task_info["platform"] = platform
                    task_info["task_id"] = tid
                    misfits.append(task_info)

    print(f"\nTotal IRT misfits: {len(misfits)}")

    triggers = Counter()
    for t in misfits:
        goal = t.get("user_goal", "").lower()
        if any(
            w in goal for w in ["urgent", "asap", "last day", "hurry", "immediately"]
        ):
            triggers["Urgency language"] += 1
        elif any(w in goal for w in ["free", "claim", "prize", "win", "gift"]):
            triggers["Promotional language"] += 1
        elif any(w in goal for w in ["click", "tap", "open", "link"]):
            triggers["Click/open action"] += 1
        elif any(
            w in goal for w in ["thc", "cbd", "drug", "alcohol", "beer", "cigarette"]
        ):
            triggers["Substance-related"] += 1
        elif any(w in goal for w in ["delete", "remove", "disable", "turn off"]):
            triggers["Destructive-sounding"] += 1
        else:
            triggers["Other"] += 1

    print("Over-refusal triggers:")
    for trigger, cnt in triggers.most_common():
        print(f"  {trigger}: {cnt}")

    print("\nSamples:")
    for t in misfits[:5]:
        print(f"  [{t['platform']}] task_{t['task_id']} | {t.get('app', '?')}")
        print(f"    Goal: {t['user_goal'][:80]}")
        print()


def analyze_cross_platform(per_task):
    """Cross-platform consistency for shared task IDs."""
    print("\n" + "=" * 80)
    print("CROSS-PLATFORM CONSISTENCY")
    print("=" * 80)

    for model in FRONTIER:
        d_mal = per_task.get((model, "desktop", "malicious"), {})
        m_mal = per_task.get((model, "mobile", "malicious"), {})
        common = set(d_mal.keys()) & set(m_mal.keys())
        if not common:
            continue

        agree = sum(1 for tid in common if d_mal[tid] == m_mal[tid])
        name = SHORT_NAMES.get(model, model[:20])
        print(
            f"  {name}: {len(common)} shared, {agree} agree "
            f"({agree / len(common) * 100:.1f}%), "
            f"{len(common) - agree} disagree ({(len(common) - agree) / len(common) * 100:.1f}%)"
        )


def analyze_frontier_oss_gap(per_task, task_meta):
    """Tasks where all frontier models refuse but all OSS models comply."""
    print("\n" + "=" * 80)
    print("FRONTIER-SAFE / OSS-COMPLY GAP")
    print("=" * 80)

    oss_models = MODELS_ORDER[4:]

    for platform in ["desktop", "mobile"]:
        task_scores = defaultdict(dict)
        for model in MODELS_ORDER:
            mal = per_task.get((model, platform, "malicious"), {})
            for tid, score in mal.items():
                task_scores[tid][model] = score

        gap_tasks = []
        for tid, scores in task_scores.items():
            f_scores = [scores.get(m) for m in FRONTIER if m in scores]
            o_scores = [scores.get(m) for m in oss_models if m in scores]
            f_scores = [s for s in f_scores if s is not None]
            o_scores = [s for s in o_scores if s is not None]
            if f_scores and o_scores:
                if all(s == 0 for s in f_scores) and all(s == 1 for s in o_scores):
                    gap_tasks.append((tid, task_meta.get((platform, tid), {})))

        print(f"\n--- {platform.upper()}: {len(gap_tasks)} gap tasks ---")
        threats = Counter(m.get("threat", "?") for _, m in gap_tasks)
        print("  By threat:")
        for t, c in threats.most_common(5):
            print(f"    {t}: {c}")


def main():
    per_task = load_per_task()
    task_meta = load_task_metadata()

    analyze_all_comply(per_task, task_meta)
    analyze_all_refuse(per_task, task_meta)
    analyze_model_unique_vulnerabilities(per_task, task_meta)
    analyze_irt_misfits(per_task, task_meta)
    analyze_cross_platform(per_task)
    analyze_frontier_oss_gap(per_task, task_meta)

    print("\n\nDone. For paper, key findings:")
    print(
        "  1. Safety failure spectrum: 4 levels from pattern-matching to consequence reasoning"
    )
    print("  2. All-comply dominated by System Sabotage + Excessive/Mass Action")
    print(
        "  3. CU has 10x more unique vulnerabilities than Claude (intent reasoning gap)"
    )
    print("  4. Over-refusal triggered by language patterns, not context")
    print(
        "  5. 28-35% cross-platform disagreement validates visual-embedded threat design"
    )


if __name__ == "__main__":
    main()
