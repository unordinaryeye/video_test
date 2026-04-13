"""tests/test_motion_gate.py — rules/motion_gate.py 단위 테스트."""

from datetime import datetime, timezone

import pytest

from rules.motion_gate import MotionGateState
from rules.loader import MotionGate as MotionGateConfig


def _config(enabled=True, idle_seconds=180, movement_threshold_px=20, classes=None):
    return MotionGateConfig(
        enabled=enabled,
        idle_seconds=float(idle_seconds),
        movement_threshold_px=float(movement_threshold_px),
        classes=classes or [],
    )


def _det(label="person", track_id=1, cx=50, cy=50):
    return {
        "label": label,
        "confidence": 0.9,
        "bbox": [cx - 5, cy - 5, cx + 5, cy + 5],
        "track_id": track_id,
    }


# ---------------------------------------------------------------------------
# enabled=False — no-op
# ---------------------------------------------------------------------------

def test_disabled_update_returns_empty():
    state = MotionGateState(_config(enabled=False))
    idle = state.update([_det()], datetime.now(timezone.utc))
    assert idle == []


def test_disabled_filter_returns_original():
    state = MotionGateState(_config(enabled=False))
    dets = [_det(track_id=1), _det(track_id=2)]
    result = state.filter(dets, [1])
    assert result is dets  # 동일 객체 반환 (no-op)


# ---------------------------------------------------------------------------
# enabled=True — update 기본 동작
# ---------------------------------------------------------------------------

def test_update_tracks_positions():
    state = MotionGateState(_config(enabled=True))
    now = datetime.now(timezone.utc)
    state.update([_det(track_id=1, cx=50, cy=50)], now)
    assert 1 in state._tracks
    assert len(state._tracks[1].positions) == 1


def test_update_accumulates_positions():
    state = MotionGateState(_config(enabled=True))
    now = datetime.now(timezone.utc)
    state.update([_det(track_id=1, cx=50, cy=50)], now)
    state.update([_det(track_id=1, cx=55, cy=55)], now)
    assert len(state._tracks[1].positions) == 2


def test_update_ignores_none_track_id():
    state = MotionGateState(_config(enabled=True))
    det = _det()
    det["track_id"] = None
    idle = state.update([det], datetime.now(timezone.utc))
    assert idle == []
    assert state._tracks == {}


def test_update_class_filter():
    state = MotionGateState(_config(enabled=True, classes=["person"]))
    dets = [_det(label="car", track_id=99)]
    state.update(dets, datetime.now(timezone.utc))
    assert 99 not in state._tracks


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------

def test_filter_removes_idle_track():
    state = MotionGateState(_config(enabled=True))
    dets = [_det(track_id=1), _det(track_id=2)]
    result = state.filter(dets, idle_ids=[1])
    assert len(result) == 1
    assert result[0]["track_id"] == 2


def test_filter_empty_idle_ids_returns_all():
    state = MotionGateState(_config(enabled=True))
    dets = [_det(track_id=1), _det(track_id=2)]
    result = state.filter(dets, idle_ids=[])
    assert result == dets


def test_filter_original_unchanged():
    state = MotionGateState(_config(enabled=True))
    dets = [_det(track_id=1), _det(track_id=2)]
    state.filter(dets, idle_ids=[1])
    assert len(dets) == 2  # 원본 불변


# ---------------------------------------------------------------------------
# idle 판정 로직
# ---------------------------------------------------------------------------

from datetime import timedelta


def _at(base: datetime, offset_seconds: float) -> datetime:
    return base + timedelta(seconds=offset_seconds)


def test_idle_detected_when_stationary_over_window():
    """idle_seconds 동안 움직임 없으면 idle로 판정."""
    cfg = _config(idle_seconds=10, movement_threshold_px=5)
    state = MotionGateState(cfg)
    base = datetime.now(timezone.utc)
    idle = []
    for i in range(12):  # 0초 ~ 11초, 1초 간격
        idle = state.update([_det(track_id=1, cx=100, cy=100)], _at(base, i))
    assert 1 in idle


def test_not_idle_when_span_too_short():
    """관측 기간이 idle_seconds 미만이면 아직 idle 아님."""
    cfg = _config(idle_seconds=10, movement_threshold_px=5)
    state = MotionGateState(cfg)
    base = datetime.now(timezone.utc)
    idle = []
    for i in range(5):  # 0초 ~ 4초 (총 4초 span)
        idle = state.update([_det(track_id=1, cx=100, cy=100)], _at(base, i))
    assert 1 not in idle


def test_not_idle_when_moving():
    """움직임이 threshold 이상이면 idle 아님."""
    cfg = _config(idle_seconds=10, movement_threshold_px=5)
    state = MotionGateState(cfg)
    base = datetime.now(timezone.utc)
    idle = []
    for i in range(15):
        # 매초 10px씩 이동
        idle = state.update([_det(track_id=1, cx=100 + i * 10, cy=100)], _at(base, i))
    assert 1 not in idle


def test_idle_exits_when_movement_resumes():
    """idle로 판정된 track이 다시 움직이면 idle 목록에서 제외."""
    cfg = _config(idle_seconds=10, movement_threshold_px=5)
    state = MotionGateState(cfg)
    base = datetime.now(timezone.utc)
    # 먼저 정지 상태 15초
    for i in range(15):
        idle = state.update([_det(track_id=1, cx=100, cy=100)], _at(base, i))
    assert 1 in idle
    # 갑자기 크게 이동
    idle = state.update([_det(track_id=1, cx=200, cy=200)], _at(base, 15))
    assert 1 not in idle


def test_idle_only_marks_still_tracks_among_many():
    """여러 track 중 정지한 것만 idle로 표시."""
    cfg = _config(idle_seconds=10, movement_threshold_px=5)
    state = MotionGateState(cfg)
    base = datetime.now(timezone.utc)
    idle = []
    for i in range(15):
        dets = [
            _det(track_id=1, cx=100, cy=100),          # 정지
            _det(track_id=2, cx=100 + i * 10, cy=100), # 움직임
        ]
        idle = state.update(dets, _at(base, i))
    assert 1 in idle
    assert 2 not in idle
