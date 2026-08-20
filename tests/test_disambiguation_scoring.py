# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for disambiguation scoring helpers."""

from disambiguation.disambiguation_grading import get_critical_score


class TestGetCriticalScore:
    def test_zero_zero(self):
        assert get_critical_score(0, 0) == 0

    def test_max_scores(self):
        assert get_critical_score(2, 2) == 4

    def test_mixed(self):
        assert get_critical_score(1, 2) == 3
