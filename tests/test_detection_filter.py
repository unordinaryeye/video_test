"""tests/test_detection_filter.py — rules/detection_filter.py 단위 테스트."""

import pytest

from rules.detection_filter import filter_detections
from rules.loader import DetectionRule


def _det(label: str, confidence: float, track_id=None):
    return {
        "label": label,
        "confidence": confidence,
        "bbox": [0.0, 0.0, 100.0, 100.0],
        "track_id": track_id,
    }


# ---------------------------------------------------------------------------
# 정상 케이스
# ---------------------------------------------------------------------------

def test_no_filter_passes_all():
    dets = [_det("person", 0.9), _det("car", 0.7)]
    rule = DetectionRule(classes=[], min_confidence=0.0)
    result = filter_detections(dets, rule)
    assert result == dets


def test_class_filter():
    dets = [_det("person", 0.9), _det("car", 0.7), _det("person", 0.6)]
    rule = DetectionRule(classes=["person"], min_confidence=0.0)
    result = filter_detections(dets, rule)
    assert all(d["label"] == "person" for d in result)
    assert len(result) == 2


def test_confidence_filter():
    dets = [_det("person", 0.3), _det("person", 0.5), _det("person", 0.8)]
    rule = DetectionRule(classes=[], min_confidence=0.5)
    result = filter_detections(dets, rule)
    assert len(result) == 2
    assert all(d["confidence"] >= 0.5 for d in result)


def test_class_and_confidence_combined():
    dets = [
        _det("person", 0.9),
        _det("person", 0.3),
        _det("car", 0.9),
    ]
    rule = DetectionRule(classes=["person"], min_confidence=0.5)
    result = filter_detections(dets, rule)
    assert len(result) == 1
    assert result[0]["confidence"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# 엣지 케이스
# ---------------------------------------------------------------------------

def test_empty_detections():
    rule = DetectionRule(classes=["person"], min_confidence=0.5)
    assert filter_detections([], rule) == []


def test_confidence_boundary_exact_threshold_passes():
    dets = [_det("person", 0.5)]
    rule = DetectionRule(classes=[], min_confidence=0.5)
    result = filter_detections(dets, rule)
    assert len(result) == 1


def test_confidence_just_below_threshold_excluded():
    dets = [_det("person", 0.4999)]
    rule = DetectionRule(classes=[], min_confidence=0.5)
    assert filter_detections(dets, rule) == []


def test_original_list_unchanged():
    dets = [_det("car", 0.9)]
    rule = DetectionRule(classes=["person"], min_confidence=0.0)
    filter_detections(dets, rule)
    assert len(dets) == 1  # 원본 불변
