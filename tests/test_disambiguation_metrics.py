# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for disambiguation metrics and get_metrics."""

from disambiguation.disambiguation_grading import get_metrics, METRIC_NAMES


class TestGetMetrics:
    def _make_result(self, precision=0.5, recall=0.5, f1=0.5, iou=0.5, delta=0.1):
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
            "iou_positive": iou,
            "delta": delta,
            "errors": [],
        }

    def test_single_result(self):
        m = get_metrics([self._make_result(precision=0.8, iou=0.6)])
        assert m["precision"] == [0.8]
        assert m["iou"] == [0.6]

    def test_multiple_results(self):
        m = get_metrics([self._make_result(precision=0.7), self._make_result(precision=0.9)])
        assert m["precision"] == [0.7, 0.9]
        assert len(m["iou"]) == 2

    def test_all_metric_names_present(self):
        m = get_metrics([self._make_result()])
        assert set(m.keys()) == set(METRIC_NAMES)

    def test_f1_extracted(self):
        m = get_metrics([self._make_result(f1=0.75)])
        assert m["f1"] == [0.75]

    def test_delta_extracted(self):
        m = get_metrics([self._make_result(delta=1.5)])
        assert m["delta"] == [1.5]
