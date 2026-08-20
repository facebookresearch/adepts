# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared utilities for ADEPTS benchmarks."""

from __future__ import annotations

import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
SAFETY_DATA_DIR = os.path.join(DATA_DIR, "safety")
SAFETY_RESULTS_DIR = os.path.join(RESULTS_DIR, "safety")
DISAMBIGUATION_DATA_DIR = os.path.join(DATA_DIR, "disambiguation")
DISAMBIGUATION_RESULTS_DIR = os.path.join(RESULTS_DIR, "disambiguation")

DEFAULT_TEST_SAMPLES = 1

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_START: float | None = None
_LOG_STEP: int = 0


def log(msg: str, *, step: bool = False) -> None:
    global _LOG_START, _LOG_STEP
    if _LOG_START is None:
        _LOG_START = time.monotonic()
    elapsed = time.monotonic() - _LOG_START
    ts = datetime.now().strftime("%H:%M:%S")
    if step:
        _LOG_STEP += 1
        print(f"[{ts} +{elapsed:.1f}s] Step {_LOG_STEP}: {msg}")
    else:
        print(f"[{ts} +{elapsed:.1f}s] {msg}")


def reset_log() -> None:
    global _LOG_START, _LOG_STEP
    _LOG_START = time.monotonic()
    _LOG_STEP = 0


# ---------------------------------------------------------------------------
# Summary Table
# ---------------------------------------------------------------------------

ACTION_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(")
STATUS_ACTION_RE = re.compile(
    r"^\s*StatusAction\(\s*(?:value\s*=\s*)?['\"]?([^)'\"]*)['\"]\s*\)"
)


@dataclass(frozen=True)
class SummaryRow:
    file_name: str
    model: str
    n: int
    n_errors: int
    accuracy: float
    accuracy_worst: float | None = None
    accuracy_best: float | None = None
    safety_categories: tuple[tuple[str, float], ...] = ()
    action_types: tuple[tuple[str, float], ...] = ()


def _is_api_error(r: dict) -> bool:
    return r.get("error") is not None or r.get("category") in ("error", "api_error")


def compute_summary_row(
    file_name: str, model: str, results: list[dict],
) -> SummaryRow | None:
    if not results:
        return None
    n = len(results)
    attempted = [r for r in results if not _is_api_error(r)]
    n_errors = n - len(attempted)
    n_attempted = len(attempted)

    has_multi_run = any(r.get("all_rewards") for r in attempted)
    accuracy_worst = None
    accuracy_best = None
    if has_multi_run:
        avg_rewards = [sum(r["all_rewards"]) / len(r["all_rewards"]) for r in attempted]
        accuracy = sum(avg_rewards) / n_attempted if n_attempted else 0.0
    else:
        correct = sum(1 for r in attempted if r.get("reward", 0.0) == 1.0)
        accuracy = correct / n_attempted if n_attempted > 0 else 0.0

    if has_multi_run:
        worst_rewards = [min(r.get("all_rewards", [r.get("reward", 0.0)])) for r in attempted]
        best_rewards = [max(r.get("all_rewards", [r.get("reward", 0.0)])) for r in attempted]
        accuracy_worst = sum(worst_rewards) / n_attempted if n_attempted else 0.0
        accuracy_best = sum(best_rewards) / n_attempted if n_attempted else 0.0

    def _parse_action_name(pred_str):
        pred_str = pred_str.strip()
        sm = STATUS_ACTION_RE.match(pred_str)
        if sm:
            return f"StatusAction({sm.group(1)})"
        am = ACTION_RE.match(pred_str)
        return am.group(1) if am else "UnknownOrEmpty"

    if has_multi_run:
        # Per-run distributions for each sample, then avg/worst/best across samples
        all_cat_names: set[str] = set()
        all_act_names: set[str] = set()
        per_sample_cats: list[list[str]] = []
        per_sample_acts: list[list[str]] = []
        for r in attempted:
            cats = r.get("all_categories", [r.get("category", "unknown")])
            acts = [_parse_action_name(p) for p in r.get("all_parsed_actions", [r.get("parsed_action", "")])]
            per_sample_cats.append(cats)
            per_sample_acts.append(acts)
            all_cat_names.update(cats)
            all_act_names.update(acts)

        k = len(per_sample_cats[0]) if per_sample_cats else 1

        # For each run index, compute the distribution across samples
        cat_per_run: list[Counter] = [Counter() for _ in range(k)]
        act_per_run: list[Counter] = [Counter() for _ in range(k)]
        for sample_cats, sample_acts in zip(per_sample_cats, per_sample_acts):
            for run_idx in range(min(k, len(sample_cats))):
                cat_per_run[run_idx][sample_cats[run_idx]] += 1
                act_per_run[run_idx][sample_acts[run_idx]] += 1

        # Compute avg/min/max rate for each category across runs
        cat_pcts_dict: dict[str, tuple[float, float, float]] = {}
        for cat in sorted(all_cat_names):
            rates = [cat_per_run[ri][cat] / n_attempted for ri in range(k)]
            cat_pcts_dict[cat] = (sum(rates) / k, min(rates), max(rates))
        cat_pcts = tuple(sorted(cat_pcts_dict.items(), key=lambda x: -x[1][0]))

        act_pcts_dict: dict[str, tuple[float, float, float]] = {}
        for act in sorted(all_act_names):
            rates = [act_per_run[ri][act] / n_attempted for ri in range(k)]
            act_pcts_dict[act] = (sum(rates) / k, min(rates), max(rates))
        act_pcts = tuple(sorted(act_pcts_dict.items(), key=lambda x: -x[1][0]))
    else:
        categories = Counter(r.get("category", "unknown") for r in attempted)
        action_types = Counter()
        for r in attempted:
            action_types[_parse_action_name(r.get("parsed_action", ""))] += 1

        cat_pcts = tuple((cat, count / n_attempted) for cat, count in categories.most_common()) if n_attempted else ()
        act_pcts = tuple((act, count / n_attempted) for act, count in action_types.most_common()) if n_attempted else ()

    return SummaryRow(
        file_name=file_name,
        model=model,
        n=n,
        n_errors=n_errors,
        accuracy=accuracy,
        accuracy_worst=accuracy_worst,
        accuracy_best=accuracy_best,
        safety_categories=cat_pcts,
        action_types=act_pcts,
    )


class SummaryPrinter:
    def __init__(self) -> None:
        self._rows: list[SummaryRow] = []
        self._all_cats: list[str] = []
        self._all_acts: list[str] = []
        self._seen_cats: set[str] = set()
        self._seen_acts: set[str] = set()
        self._has_multi_run: bool = False
        self._has_errors: bool = False

    def _update_columns(self, row: SummaryRow) -> None:
        if row.accuracy_worst is not None:
            self._has_multi_run = True
        if row.n_errors > 0:
            self._has_errors = True
        for name, _ in row.safety_categories:
            if name not in self._seen_cats:
                self._seen_cats.add(name)
                self._all_cats.append(name)
                self._all_cats.sort()
        for name, _ in row.action_types:
            if name not in self._seen_acts:
                self._seen_acts.add(name)
                self._all_acts.append(name)
                self._all_acts.sort()

    def _triplet(self, v, default=(0.0, 0.0, 0.0)):
        """Extract (worst, avg, best) from a value that's either a float or (avg, worst, best) tuple."""
        if isinstance(v, tuple):
            avg, worst, best = v
            return worst, avg, best
        return v, v, v

    def _row_cells(self, row: SummaryRow) -> list[str]:
        cat_map = dict(row.safety_categories)
        act_map = dict(row.action_types)
        short = row.file_name.replace("safety_", "").replace(".jsonl", "")
        cells = [row.model, short, str(row.n)]
        if self._has_errors:
            cells.append(str(row.n_errors) if row.n_errors > 0 else "")
        if self._has_multi_run:
            w = f"{row.accuracy_worst:.1%}" if row.accuracy_worst is not None else ""
            b = f"{row.accuracy_best:.1%}" if row.accuracy_best is not None else ""
            cells += [w, f"{row.accuracy:.1%}", b]
            for group_list, all_names in [(cat_map, self._all_cats), (act_map, self._all_acts)]:
                cells.append("|")
                for name in all_names:
                    worst, avg, best = self._triplet(group_list.get(name, (0.0, 0.0, 0.0)))
                    cells += [f"{worst:.0%}", f"{avg:.0%}", f"{best:.0%}"]
        else:
            cells.append(f"{row.accuracy:.1%}")
            cells.append("|")
            cells += [f"{cat_map.get(c, 0.0):.1%}" for c in self._all_cats]
            cells.append("|")
            cells += [f"{act_map.get(a, 0.0):.1%}" for a in self._all_acts]
        return cells

    def _render_table(self) -> str:
        if self._has_multi_run:
            return self._render_multi_run_table()
        return self._render_single_run_table()

    @staticmethod
    def _compute_widths(all_rows):
        n_cols = max(len(row) for row in all_rows)
        widths = [0] * n_cols
        for row in all_rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))
        return widths, n_cols

    @staticmethod
    def _fmt(cols, widths, align="left"):
        parts = []
        for i, c in enumerate(cols):
            w = widths[i] if i < len(widths) else len(c)
            if align == "center":
                parts.append(c.center(w))
            elif align == "right":
                parts.append(c.rjust(w))
            else:
                parts.append(c.ljust(w))
        return "  ".join(parts)

    def _render_single_run_table(self) -> str:
        headers = ["model", "file", "n"]
        if self._has_errors:
            headers.append("err")
        headers.append("accuracy")
        sep_idx1 = len(headers)
        headers += ["|"] + self._all_cats
        sep_idx2 = len(headers)
        headers += ["|"] + self._all_acts

        all_cells = [headers] + [self._row_cells(r) for r in self._rows]
        widths, n_cols = self._compute_widths(all_cells)
        sep_idxs = {sep_idx1, sep_idx2}
        divider = ["|" if i in sep_idxs else "-" * widths[i] for i in range(n_cols)]

        lines = [self._fmt(headers, widths)]
        lines.append(self._fmt(divider, widths))
        lines += [self._fmt(c, widths) for c in all_cells[1:]]
        return "\n".join(lines)

    def _render_multi_run_table(self) -> str:
        fixed = ["model", "file", "n"]
        if self._has_errors:
            fixed.append("err")
        subs = ["worst", "avg", "best"]

        # Build two header rows
        h1 = [""] * len(fixed)
        h2 = list(fixed)
        sep_positions: set[int] = set()

        # Accuracy group
        h1 += ["", "accuracy", ""]
        h2 += subs

        # Category groups
        for metric_names in [self._all_cats, self._all_acts]:
            if metric_names:
                pos = len(h1)
                sep_positions.add(pos)
                h1.append("|")
                h2.append("|")
                for name in metric_names:
                    h1 += ["", name, ""]
                    h2 += subs

        all_data = [self._row_cells(r) for r in self._rows]
        widths, n_cols = self._compute_widths([h1, h2] + all_data)
        divider = ["|" if i in sep_positions else "-" * widths[i] for i in range(n_cols)]

        lines = [self._fmt(h1, widths, "center"), self._fmt(h2, widths, "center"), self._fmt(divider, widths)]
        lines += [self._fmt(c, widths, "right") for c in all_data]
        return "\n".join(lines)

    def add_row(self, row: SummaryRow) -> None:
        self._update_columns(row)
        self._rows.append(row)

    def print_table(self) -> None:
        if self._rows:
            print(self._render_table())
        else:
            print("No results found.")
