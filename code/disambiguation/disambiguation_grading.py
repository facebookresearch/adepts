# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Evaluation, scoring, metrics, and plotting for the disambiguation benchmark.

Implements the disambiguation evaluation from the paper: per-item semantic
matching feeds precision/recall/F1 and the Severity Calibration Error (delta)
defined in §3.4 (computed in common.compute_disambig_prf1_delta, Eq. 1). The
obviousness/consequence rubric (each 0-2) is from §3.3; IoU / IoU-positive are
the additional overlap metrics from Appendix G.
"""

from __future__ import annotations

import math
import os
import sys
import traceback
from collections import Counter
from enum import auto, Enum
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analytics"))
from common import compute_disambig_prf1_delta


# ---------------------------------------------------------------------------
# Enums and Configuration
# ---------------------------------------------------------------------------

class BenchmarkType(Enum):
    DESKTOP = "desktop"
    MOBILE = "mobile"


class PromptType(Enum):
    WITH_SCORE = auto()
    NO_SCORE = auto()


BENCHMARK_CONFIGS = {
    BenchmarkType.DESKTOP: {
        "dataset_file_name": "tasks_desktop.json",
        "scored_file_name": "desktop_scored.json",
    },
    BenchmarkType.MOBILE: {
        "dataset_file_name": "tasks_mobile.json",
        "scored_file_name": "mobile_scored.json",
    },
}

METRIC_NAMES = [
    "precision", "recall", "f1",
    "iou", "iou_positive", "delta",
]


# ---------------------------------------------------------------------------
# Scoring Helpers
# ---------------------------------------------------------------------------

def get_critical_score(obviousness_score, consequence_score):
    # Combines the obviousness and consequence axes (each 0-2; paper §3.3) into a
    # single severity used to decide whether a clarification is "critical" enough
    # to count (thresholded against score_threshold in the WITH_SCORE path).
    return obviousness_score + consequence_score


def get_metrics(results_array):
    def _extract(key):
        return [item[key] for item in results_array]

    return {name: _extract(name) for name in METRIC_NAMES}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_and_filter_sample(sample_index, benchmark_generations, score_threshold, model, mode, get_key_fn):
    try:
        key = get_key_fn(model.value, mode, sample_index)
        if key in benchmark_generations:
            gen_clarifications = benchmark_generations[key]["gen_clarifications"]
            error_type = benchmark_generations[key]["error_type"]
        else:
            raise ValueError("Key not found in benchmark_generations")

        high_gen_clarifications = gen_clarifications
        if mode == PromptType.WITH_SCORE:
            high_gen_clarifications = []
            for c in gen_clarifications:
                # A model may return a clarification without score fields (or a
                # null question). Skip those instead of hard-indexing the keys,
                # which would KeyError (missing) or TypeError (None + None).
                if c.get("question") is None:
                    continue
                obviousness_score = c.get("obviousness_score")
                consequence_score = c.get("consequence_score")
                if obviousness_score is None or consequence_score is None:
                    continue
                gen_critical_score = get_critical_score(obviousness_score, consequence_score)
                if gen_critical_score >= score_threshold:
                    question_str = str(c["question"]).lower()
                    if question_str != "null" and question_str != "none":
                        high_gen_clarifications.append(c)
        gen_should_disambiguate = len(high_gen_clarifications) > 0

        return (gen_should_disambiguate, high_gen_clarifications, error_type)
    except Exception as e:
        print(f"  Error in sample {sample_index} for {model.value}: {type(e).__name__}: {e}")
        return (False, [], str(type(e).__name__))


async def benchmark_evaluation(benchmark_dataset, benchmark_generations, model, mode, score_ground_threshold, get_key_fn):
    errors = []
    iou_results = []
    iou_positive_results = []
    pairs = []

    for i, sample in enumerate(benchmark_dataset):
        try:
            (gen_should_disambiguate, high_gen_clarifications, error_type) = evaluate_and_filter_sample(
                i, benchmark_generations, score_ground_threshold, model, mode, get_key_fn
            )

            ground_clarifications = sample["clarifications"]
            high_ground_clarifications = [
                c for c in ground_clarifications
                if c["obviousness_score"] is not None
                and c["consequence_score"] is not None
                and get_critical_score(int(c["obviousness_score"]), int(c["consequence_score"])) >= score_ground_threshold
            ]
            if ground_clarifications is None or ground_clarifications == "None" or ground_clarifications == []:
                ground_should_disambiguate = False
            else:
                ground_should_disambiguate = any(
                    gc["obviousness_score"] is not None
                    and gc["consequence_score"] is not None
                    and get_critical_score(gc["obviousness_score"], gc["consequence_score"]) >= score_ground_threshold
                    for gc in ground_clarifications
                )

            correct_generation = gen_should_disambiguate == ground_should_disambiguate and error_type is None

            pairs.append((high_gen_clarifications, high_ground_clarifications))

            # --- IoU (per-sample) ---
            gt_questions = set(g["question"] for g in high_ground_clarifications)
            match_count = sum(
                1 for g in high_gen_clarifications
                if g.get("match") and g.get("match_ground_text") in gt_questions
            )
            all_params = set()
            redundant_elements = set()

            for ground in high_ground_clarifications:
                all_params.add(ground["question"])
            for gen in high_gen_clarifications:
                gen_question_text = gen.get("question", "")
                all_params.add(gen_question_text)
                if gen.get("match") and gen.get("match_ground_text") in gt_questions:
                    if gen_question_text != gen.get("match_ground_text"):
                        redundant_elements.add(gen.get("match_ground_text"))

            unique_elements = all_params - redundant_elements
            iou = 0
            if correct_generation:
                if len(unique_elements) > 0:
                    iou = match_count / len(unique_elements)
                    iou_positive_results.append(iou)
                elif len(high_ground_clarifications) == 0 and len(high_gen_clarifications) == 0:
                    iou = 1
            else:
                iou_positive_results.append(iou)

            iou_results.append(iou)

            if error_type:
                errors.append(error_type)

        except Exception as e:
            print(f"    Error in evaluation: {type(e).__name__}")
            traceback.print_exc()
            errors.append(str(type(e).__name__))

    prf = compute_disambig_prf1_delta(pairs)

    return {
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        "iou": sum(iou_results) / len(iou_results) if iou_results else 0,
        "iou_positive": sum(iou_positive_results) / len(iou_positive_results) if iou_positive_results else 0,
        "delta": prf["delta"],
        "errors": errors,
    }


def analyze_dataset(benchmark_dataset):
    num_samples = len(benchmark_dataset)
    clarification_types = []
    clarification_scores = []
    no_clarification_count = 0
    missing_score_count = 0
    for sample in benchmark_dataset:
        clarifications = sample.get("clarifications", [])
        if not clarifications:
            no_clarification_count += 1
        else:
            for clarification in clarifications:
                clarification_types.append(clarification.get("type", ""))
                obviousness = clarification.get("obviousness_score")
                consequence = clarification.get("consequence_score")
                if obviousness is None or consequence is None:
                    missing_score_count += 1
                    continue
                clarification_scores.append(get_critical_score(obviousness, consequence))
    type_histogram = Counter(clarification_types)
    score_histogram = Counter(clarification_scores)
    print(f"  Total samples: {num_samples}")
    print(f"  Samples with no clarifications: {no_clarification_count}")
    print(f"  Clarifications with missing scores: {missing_score_count}")
    print("\n  Histogram of clarification types:")
    for k, v in type_histogram.items():
        print(f"    {k if k else '[empty]'}: {v}")
    print("\n  Histogram of score types:")
    for k, v in score_histogram.items():
        print(f"    {k}: {v}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_model_histogram(model_counts, ax, title="", xlabel=None, ylabel="Value", color="skyblue", edgecolor="black"):
    model_count_pairs = [
        (m, model_counts[m][0] if isinstance(model_counts[m], list) else model_counts[m])
        for m in model_counts
    ]
    sorted_pairs = sorted(model_count_pairs, key=lambda x: x[1] if x[1] is not None and x[1] == x[1] else 0, reverse=True)
    models_list = [m for m, c in sorted_pairs]
    counts = [c if c is not None and c == c else 0 for m, c in sorted_pairs]
    ax.bar(models_list, counts, color=color, edgecolor=edgecolor)
    ax.set_title(title)
    ax.set_xlabel(xlabel or "")
    ax.set_ylabel(ylabel or "")
    ax.set_xticks(range(len(models_list)))
    ax.set_xticklabels(models_list, rotation=45, ha="right", fontsize=8)
    max_count = max(counts) if counts else 0
    if max_count > 1:
        ax.set_ylim(0, max_count)
    else:
        ax.set_ylim(0, 1)


def plot_all(metrics_by_name):
    all_labels = sorted({lbl for sd in metrics_by_name.values() for lbl in sd})
    print("=" * 80)
    print("Metrics results (per threshold index)")
    print("=" * 80)
    for metric_name, series_dict in metrics_by_name.items():
        print(f"\n{metric_name.replace('_', ' ').title()}:")
        for label in sorted(series_dict.keys()):
            vals = ", ".join(f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for v in series_dict[label])
            print(f"  {label}: [{vals}]")
    print("=" * 80)

    cmap = matplotlib.colormaps["tab20"].resampled(len(all_labels))
    label_to_color = {label: cmap(i) for i, label in enumerate(all_labels)}
    num_plots = len(metrics_by_name)
    n_cols = 6
    n_rows = math.ceil(num_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True)
    axes = axes.flatten()
    for idx, (ax, (metric_name, series_dict)) in enumerate(zip(axes, metrics_by_name.items())):
        for label, values in series_dict.items():
            ax.plot(range(len(values)), values, marker="o", label=label, color=label_to_color[label])
        ax.set_xlabel("Threshold Index")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name, fontsize=16)
        ax.grid(True)
        if idx == num_plots - 1:
            handles, labels = ax.get_legend_handles_labels()
            sorted_pairs = sorted(zip(labels, handles), key=lambda x: x[0])
            if sorted_pairs:
                sorted_labels, sorted_handles = zip(*sorted_pairs)
                ax.legend(sorted_handles, sorted_labels, loc="center left", bbox_to_anchor=(1, 0.5))
    for ax in axes[num_plots:]:
        ax.set_visible(False)
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_all_histograms(metrics_by_name_no_threshold):
    metrics = list(metrics_by_name_no_threshold.items())
    n_metrics = len(metrics)
    print("=" * 80)
    print("Histogram metrics results")
    print("=" * 80)
    for metric_name, model_counts in metrics:
        print(f"\n{metric_name.replace('_', ' ').title()}:")
        for model in sorted(model_counts.keys()):
            value = model_counts[model]
            print(f"  {model}: {value:.4f}" if isinstance(value, (int, float)) else f"  {model}: {value}")
    print("=" * 80)

    n_cols = 4
    n_rows = math.ceil(n_metrics / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), squeeze=False)
    for idx, (metric_name, model_counts) in enumerate(metrics):
        ax = axes[idx // n_cols][idx % n_cols]
        plot_model_histogram(model_counts, ax, title=metric_name.replace("_", " ").title(), xlabel=None, ylabel=metric_name.replace("_", " ").title())
    for idx in range(n_metrics, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")
    plt.tight_layout()
    plt.close(fig)
    return fig


def save_figure_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return buf.getvalue()
