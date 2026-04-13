"""폴리곤 내부 판정 및 zone 이벤트 생성."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Tuple

from rules.loader import Zone, ZoneRule


@dataclass
class ZoneEvent:
    zone_name: str
    rule_type: str
    detections: List[dict]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """Ray casting 알고리즘으로 점이 폴리곤 내부에 있는지 판정한다.

    외부 의존성 없이 순수 파이썬으로 구현.

    Args:
        point: (x, y) 좌표.
        polygon: [[x1, y1], [x2, y2], ...] 형식의 꼭짓점 목록 (최소 3개).

    Returns:
        내부(경계 포함)이면 True, 외부면 False.
    """
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i][0], polygon[i][1]
        xj, yj = polygon[j][0], polygon[j][1]

        # 경계선 위의 점 처리
        if yi == yj:
            # 수평 선분: 점이 선분 위에 있으면 내부로 간주
            if yi == y and min(xi, xj) <= x <= max(xi, xj):
                return True
        else:
            if min(yi, yj) <= y <= max(yi, yj):
                # 교차점의 x 좌표
                t = (y - yi) / (yj - yi)
                intersect_x = xi + t * (xj - xi)
                if x == intersect_x:
                    # 점이 경계선 위에 있음
                    return True
                if x < intersect_x and min(yi, yj) < y <= max(yi, yj):
                    inside = not inside

        j = i

    return inside


def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """bbox [x1, y1, x2, y2]의 중심점을 반환한다."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _apply_zone_rule(
    detections: List[dict],
    rule: ZoneRule,
    polygon: List[List[float]],
) -> List[dict]:
    """zone rule에 해당하는 detection 목록을 반환한다."""
    candidates = [
        det for det in detections
        if not rule.classes or det.get("label", "") in rule.classes
    ]

    if rule.type == "entry":
        return [
            det for det in candidates
            if point_in_polygon(_bbox_center(det.get("bbox", [0, 0, 0, 0])), polygon)
        ]

    if rule.type == "count_exceeds":
        inside = [
            det for det in candidates
            if point_in_polygon(_bbox_center(det.get("bbox", [0, 0, 0, 0])), polygon)
        ]
        return inside if len(inside) > rule.threshold else []

    # TODO: "dwell_time", "speed_exceeds" 등 추가 룰 타입 구현
    return []


def check_zones(
    detections: List[dict],
    zones: List[Zone],
) -> List[ZoneEvent]:
    """각 zone의 각 rule을 평가하여 ZoneEvent 목록을 반환한다.

    Args:
        detections: filter_detections를 통과한 detection 목록.
        zones: 설정에서 로드한 Zone 목록.

    Returns:
        발생한 ZoneEvent 목록. 이벤트가 없으면 빈 리스트.
    """
    events: List[ZoneEvent] = []
    now = datetime.now(timezone.utc)

    for zone in zones:
        for rule in zone.rules:
            matched = _apply_zone_rule(detections, rule, zone.polygon)
            if matched:
                events.append(
                    ZoneEvent(
                        zone_name=zone.name,
                        rule_type=rule.type,
                        detections=matched,
                        timestamp=now,
                    )
                )

    return events
