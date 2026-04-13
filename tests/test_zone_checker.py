"""tests/test_zone_checker.py — rules/zone_checker.py 단위 테스트."""

import pytest

from rules.zone_checker import ZoneEvent, check_zones, point_in_polygon
from rules.loader import Zone, ZoneRule


# ---------------------------------------------------------------------------
# point_in_polygon
# ---------------------------------------------------------------------------

SQUARE = [[0, 0], [100, 0], [100, 100], [0, 100]]


def test_point_inside_square():
    assert point_in_polygon((50, 50), SQUARE) is True


def test_point_outside_square():
    assert point_in_polygon((150, 150), SQUARE) is False


def test_point_on_boundary():
    # 경계선 위 점은 내부로 간주
    assert point_in_polygon((50, 0), SQUARE) is True


def test_point_on_corner():
    assert point_in_polygon((0, 0), SQUARE) is True


def test_point_just_outside():
    assert point_in_polygon((100.1, 50), SQUARE) is False


def test_triangle_inside():
    triangle = [[0, 0], [100, 0], [50, 100]]
    assert point_in_polygon((50, 40), triangle) is True


def test_triangle_outside():
    triangle = [[0, 0], [100, 0], [50, 100]]
    assert point_in_polygon((5, 90), triangle) is False


# ---------------------------------------------------------------------------
# check_zones — entry 룰
# ---------------------------------------------------------------------------

def _det(label, cx=50, cy=50, track_id=None):
    """중심이 (cx, cy)인 detection."""
    return {
        "label": label,
        "confidence": 0.9,
        "bbox": [cx - 5, cy - 5, cx + 5, cy + 5],
        "track_id": track_id,
    }


def _zone(name, rules, polygon=None):
    poly = polygon or SQUARE
    return Zone(name=name, polygon=poly, rules=rules)


def test_entry_rule_inside_generates_event():
    zone = _zone("Z", [ZoneRule(type="entry", classes=["person"])])
    dets = [_det("person", cx=50, cy=50)]
    events = check_zones(dets, [zone])
    assert len(events) == 1
    assert events[0].zone_name == "Z"
    assert events[0].rule_type == "entry"


def test_entry_rule_outside_no_event():
    zone = _zone("Z", [ZoneRule(type="entry", classes=["person"])])
    dets = [_det("person", cx=200, cy=200)]
    assert check_zones(dets, [zone]) == []


def test_entry_rule_class_filter():
    zone = _zone("Z", [ZoneRule(type="entry", classes=["person"])])
    dets = [_det("car", cx=50, cy=50)]
    assert check_zones(dets, [zone]) == []


def test_entry_rule_no_class_filter_all_pass():
    zone = _zone("Z", [ZoneRule(type="entry", classes=[])])
    dets = [_det("car", cx=50, cy=50)]
    events = check_zones(dets, [zone])
    assert len(events) == 1


# ---------------------------------------------------------------------------
# check_zones — count_exceeds 룰
# ---------------------------------------------------------------------------

def test_count_exceeds_above_threshold():
    zone = _zone("Z", [ZoneRule(type="count_exceeds", threshold=2, classes=["person"])])
    dets = [_det("person", cx=50, cy=50) for _ in range(3)]
    events = check_zones(dets, [zone])
    assert len(events) == 1
    assert len(events[0].detections) == 3


def test_count_exceeds_at_threshold_no_event():
    zone = _zone("Z", [ZoneRule(type="count_exceeds", threshold=2, classes=["person"])])
    dets = [_det("person", cx=50, cy=50) for _ in range(2)]
    assert check_zones(dets, [zone]) == []


def test_count_exceeds_below_threshold_no_event():
    zone = _zone("Z", [ZoneRule(type="count_exceeds", threshold=2, classes=["person"])])
    dets = [_det("person", cx=50, cy=50)]
    assert check_zones(dets, [zone]) == []


# ---------------------------------------------------------------------------
# 엣지 케이스
# ---------------------------------------------------------------------------

def test_empty_detections():
    zone = _zone("Z", [ZoneRule(type="entry")])
    assert check_zones([], [zone]) == []


def test_empty_zones():
    dets = [_det("person")]
    assert check_zones(dets, []) == []


def test_event_has_timestamp():
    zone = _zone("Z", [ZoneRule(type="entry")])
    dets = [_det("person")]
    events = check_zones(dets, [zone])
    assert events[0].timestamp is not None
