"""정지 상태 게이팅 — track_id 기반."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, List, Optional, Tuple

from rules.loader import MotionGate as MotionGateConfig


@dataclass
class _TrackHistory:
    """단일 track의 위치 히스토리."""
    positions: Deque[Tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=5000)
    )
    # 각 항목: (x, y, timestamp_seconds)


class MotionGateState:
    """track_id별 최근 위치 히스토리를 관리하고 idle 여부를 판정한다.

    enabled=False이면 모든 메서드가 no-op이다.
    """

    def __init__(self, config: MotionGateConfig) -> None:
        self._config = config
        self._tracks: Dict[int, _TrackHistory] = {}

    def _bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(
        self,
        detections: List[dict],
        now: Optional[datetime] = None,
    ) -> List[int]:
        """위치 히스토리를 갱신하고 idle로 판정된 track_id 목록을 반환한다.

        idle 조건 (둘 다 충족 시):
          - idle_seconds 동안 연속 관측됨 (window 내 첫/마지막 시간차 >= idle_seconds)
          - window 내 위치들의 bounding box 대각선 < movement_threshold_px
        """
        if not self._config.enabled:
            return []

        ts = now.timestamp() if now else datetime.utcnow().timestamp()

        for det in detections:
            track_id = det.get("track_id")
            if track_id is None:
                continue

            label = det.get("label", "")
            if self._config.classes and label not in self._config.classes:
                continue

            bbox = det.get("bbox", [0, 0, 0, 0])
            cx, cy = self._bbox_center(bbox)

            if track_id not in self._tracks:
                self._tracks[track_id] = _TrackHistory()
            self._tracks[track_id].positions.append((cx, cy, ts))

        cutoff = ts - self._config.idle_seconds
        idle_ids: List[int] = []

        for track_id, hist in self._tracks.items():
            while hist.positions and hist.positions[0][2] < cutoff:
                hist.positions.popleft()

            if len(hist.positions) < 2:
                continue

            span = hist.positions[-1][2] - hist.positions[0][2]
            if span < self._config.idle_seconds:
                continue

            xs = [p[0] for p in hist.positions]
            ys = [p[1] for p in hist.positions]
            dx = max(xs) - min(xs)
            dy = max(ys) - min(ys)
            displacement = (dx * dx + dy * dy) ** 0.5

            if displacement < self._config.movement_threshold_px:
                idle_ids.append(track_id)

        return idle_ids

    def filter(
        self,
        detections: List[dict],
        idle_ids: List[int],
    ) -> List[dict]:
        """idle로 판정된 track을 detection 목록에서 제거한다.

        Args:
            detections: detection dict 목록.
            idle_ids: 제거할 track_id 목록 (update()의 반환값).

        Returns:
            idle track이 제거된 새 리스트 (원본 불변).
            enabled=False이면 detections를 그대로 반환.
        """
        if not self._config.enabled:
            return detections

        if not idle_ids:
            return detections

        idle_set = set(idle_ids)
        return [
            det for det in detections
            if det.get("track_id") not in idle_set
        ]
