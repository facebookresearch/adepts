"""Unit tests for bounding box hit detection."""

from safety.safety_benchmark import _in_any_bbox

BBOX = [{"points": [
    {"xNorm": 0.4, "yNorm": 0.4},
    {"xNorm": 0.6, "yNorm": 0.4},
    {"xNorm": 0.6, "yNorm": 0.6},
    {"xNorm": 0.4, "yNorm": 0.6},
]}]


class TestInAnyBbox:
    def test_inside(self):
        assert _in_any_bbox(0.5, 0.5, BBOX) is True

    def test_outside(self):
        assert _in_any_bbox(0.1, 0.1, BBOX) is False

    def test_on_edge(self):
        assert _in_any_bbox(0.4, 0.4, BBOX) is True

    def test_multiple_bboxes_hit_second(self):
        bbox2 = [
            {"points": [{"xNorm": 0.0, "yNorm": 0.0}, {"xNorm": 0.1, "yNorm": 0.1}]},
            {"points": [{"xNorm": 0.8, "yNorm": 0.8}, {"xNorm": 0.9, "yNorm": 0.9}]},
        ]
        assert _in_any_bbox(0.85, 0.85, bbox2) is True

    def test_empty_points(self):
        assert _in_any_bbox(0.5, 0.5, [{"points": []}]) is False

    def test_missing_keys(self):
        assert _in_any_bbox(0.5, 0.5, [{"points": [{"x": 0.4, "y": 0.4}]}]) is False
