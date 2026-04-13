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
