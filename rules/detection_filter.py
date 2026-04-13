"""클래스 및 신뢰도 기반 detection 필터."""

from __future__ import annotations

from typing import List

from rules.loader import DetectionRule


def filter_detections(
    detections: List[dict],
    rule: DetectionRule,
) -> List[dict]:
    """DetectionRule에 따라 detection 목록을 필터링한다.

    Args:
        detections: detection dict 목록.
            각 항목 형식: {"label": str, "confidence": float,
                          "bbox": [x1, y1, x2, y2], "track_id": int | None}
        rule: 필터링 기준 룰.

    Returns:
        필터를 통과한 detection 목록 (새 리스트, 원본 불변).
    """
    result = []
    for det in detections:
        label = det.get("label", "")
        confidence = det.get("confidence", 0.0)

        if rule.classes and label not in rule.classes:
            continue

        if confidence < rule.min_confidence:
            continue

        result.append(det)

    return result
