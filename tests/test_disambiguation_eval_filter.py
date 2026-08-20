"""Unit tests for disambiguation evaluate_and_filter_sample."""

from disambiguation.disambiguation_grading import evaluate_and_filter_sample, PromptType


class _MockModel:
    def __init__(self, value):
        self.value = value


def _get_key(model, mode, idx):
    return f"{model}_{mode.name}_{idx}"


class TestEvaluateAndFilterSample:
    def test_no_score_mode_returns_all_clarifications(self):
        gens = {
            "test_NO_SCORE_0": {
                "gen_clarifications": [
                    {"question": "Which color?"},
                    {"question": "Which size?"},
                ],
                "error_type": None,
            }
        }
        should, clarifs, err = evaluate_and_filter_sample(
            0, gens, 0, _MockModel("test"), PromptType.NO_SCORE, _get_key
        )
        assert should is True
        assert len(clarifs) == 2
        assert err is None

    def test_with_score_filters_below_threshold(self):
        gens = {
            "test_WITH_SCORE_0": {
                "gen_clarifications": [
                    {"question": "Which color?", "obviousness_score": 0, "consequence_score": 0},
                    {"question": "Which size?", "obviousness_score": 2, "consequence_score": 2},
                ],
                "error_type": None,
            }
        }
        should, clarifs, err = evaluate_and_filter_sample(
            0, gens, 3, _MockModel("test"), PromptType.WITH_SCORE, _get_key
        )
        assert should is True
        assert len(clarifs) == 1
        assert clarifs[0]["question"] == "Which size?"

    def test_with_score_all_below_threshold(self):
        gens = {
            "test_WITH_SCORE_0": {
                "gen_clarifications": [
                    {"question": "Which color?", "obviousness_score": 0, "consequence_score": 0},
                ],
                "error_type": None,
            }
        }
        should, clarifs, err = evaluate_and_filter_sample(
            0, gens, 2, _MockModel("test"), PromptType.WITH_SCORE, _get_key
        )
        assert should is False
        assert len(clarifs) == 0

    def test_empty_clarifications(self):
        gens = {
            "test_NO_SCORE_0": {
                "gen_clarifications": [],
                "error_type": None,
            }
        }
        should, clarifs, err = evaluate_and_filter_sample(
            0, gens, 0, _MockModel("test"), PromptType.NO_SCORE, _get_key
        )
        assert should is False
        assert len(clarifs) == 0

    def test_missing_key_returns_error(self):
        should, clarifs, err = evaluate_and_filter_sample(
            0, {}, 0, _MockModel("test"), PromptType.NO_SCORE, _get_key
        )
        assert should is False
        assert err is not None

    def test_null_question_filtered_in_with_score(self):
        gens = {
            "test_WITH_SCORE_0": {
                "gen_clarifications": [
                    {"question": None, "obviousness_score": 2, "consequence_score": 2},
                ],
                "error_type": None,
            }
        }
        should, clarifs, err = evaluate_and_filter_sample(
            0, gens, 0, _MockModel("test"), PromptType.WITH_SCORE, _get_key
        )
        assert should is False

    def test_error_type_preserved(self):
        gens = {
            "test_NO_SCORE_0": {
                "gen_clarifications": [{"question": "Which?"}],
                "error_type": "TimeoutError",
            }
        }
        should, clarifs, err = evaluate_and_filter_sample(
            0, gens, 0, _MockModel("test"), PromptType.NO_SCORE, _get_key
        )
        assert should is True
        assert err == "TimeoutError"
