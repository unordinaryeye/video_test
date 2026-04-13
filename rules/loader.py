"""YAML 룰 로드 및 스키마 검증."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


@dataclass
class DetectionRule:
    classes: List[str] = field(default_factory=list)
    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.min_confidence <= 1.0):
            raise ValueError(
                f"detection.min_confidence must be between 0.0 and 1.0, got {self.min_confidence}"
            )


@dataclass
class ZoneRule:
    type: str = "entry"
    classes: List[str] = field(default_factory=list)
    threshold: int = 1

    def __post_init__(self) -> None:
        supported = {"entry", "count_exceeds"}
        if self.type not in supported:
            raise ValueError(
                f"zones[].rules[].type must be one of {supported}, got '{self.type}'"
            )
        if self.threshold < 1:
            raise ValueError(
                f"zones[].rules[].threshold must be >= 1, got {self.threshold}"
            )


@dataclass
class Zone:
    name: str
    polygon: List[List[float]]
    rules: List[ZoneRule] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("zones[].name must not be empty")
        if len(self.polygon) < 3:
            raise ValueError(
                f"Zone '{self.name}': polygon must have at least 3 points, got {len(self.polygon)}"
            )
        for i, pt in enumerate(self.polygon):
            if len(pt) != 2:
                raise ValueError(
                    f"Zone '{self.name}': polygon[{i}] must be [x, y], got {pt}"
                )


@dataclass
class MotionGate:
    enabled: bool = False
    idle_seconds: float = 180.0
    movement_threshold_px: float = 20.0
    classes: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.idle_seconds <= 0:
            raise ValueError(
                f"motion_gate.idle_seconds must be > 0, got {self.idle_seconds}"
            )
        if self.movement_threshold_px < 0:
            raise ValueError(
                f"motion_gate.movement_threshold_px must be >= 0, got {self.movement_threshold_px}"
            )


@dataclass
class Rules:
    detection: DetectionRule = field(default_factory=DetectionRule)
    zones: List[Zone] = field(default_factory=list)
    motion_gate: MotionGate = field(default_factory=MotionGate)


def _parse_detection_rule(data: dict) -> DetectionRule:
    classes = data.get("classes", [])
    if not isinstance(classes, list):
        raise ValueError(f"detection.classes must be a list, got {type(classes).__name__}")
    min_conf = data.get("min_confidence", 0.0)
    if not isinstance(min_conf, (int, float)):
        raise ValueError(
            f"detection.min_confidence must be a number, got {type(min_conf).__name__}"
        )
    return DetectionRule(classes=[str(c) for c in classes], min_confidence=float(min_conf))


def _parse_zone_rule(data: dict, zone_name: str, idx: int) -> ZoneRule:
    if "type" not in data:
        raise ValueError(f"Zone '{zone_name}': rules[{idx}] missing required field 'type'")
    rule_type = data["type"]
    classes = data.get("classes", [])
    if not isinstance(classes, list):
        raise ValueError(
            f"Zone '{zone_name}': rules[{idx}].classes must be a list"
        )
    threshold = data.get("threshold", 1)
    if not isinstance(threshold, int):
        raise ValueError(
            f"Zone '{zone_name}': rules[{idx}].threshold must be an integer"
        )
    return ZoneRule(type=rule_type, classes=[str(c) for c in classes], threshold=threshold)


def _parse_zone(data: dict, idx: int) -> Zone:
    if "name" not in data:
        raise ValueError(f"zones[{idx}] missing required field 'name'")
    if "polygon" not in data:
        raise ValueError(f"zones[{idx}] missing required field 'polygon'")
    name = data["name"]
    polygon = data["polygon"]
    if not isinstance(polygon, list):
        raise ValueError(f"Zone '{name}': polygon must be a list")
    rules_data = data.get("rules", [])
    if not isinstance(rules_data, list):
        raise ValueError(f"Zone '{name}': rules must be a list")
    rules = [_parse_zone_rule(r, name, i) for i, r in enumerate(rules_data)]
    return Zone(name=name, polygon=polygon, rules=rules)


def _parse_motion_gate(data: dict) -> MotionGate:
    enabled = data.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(
            f"motion_gate.enabled must be a boolean, got {type(enabled).__name__}"
        )
    idle_seconds = data.get("idle_seconds", 180.0)
    movement_threshold_px = data.get("movement_threshold_px", 20.0)
    classes = data.get("classes", [])
    if not isinstance(classes, list):
        raise ValueError("motion_gate.classes must be a list")
    return MotionGate(
        enabled=enabled,
        idle_seconds=float(idle_seconds),
        movement_threshold_px=float(movement_threshold_px),
        classes=[str(c) for c in classes],
    )


def load_rules(path: Optional[str]) -> Rules:
    """YAML 파일에서 룰을 로드한다. 파일이 없으면 기본값(전부 통과)을 반환한다.

    Args:
        path: YAML 파일 경로. None이거나 파일이 없으면 기본 Rules 반환.

    Returns:
        파싱된 Rules 객체.

    Raises:
        ValueError: 스키마가 잘못된 경우. 어느 필드가 잘못됐는지 메시지에 포함.
    """
    if path is None or not os.path.exists(path):
        return Rules()

    with open(path, "r", encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML file '{path}': {e}") from e

    if raw is None:
        return Rules()

    if not isinstance(raw, dict):
        raise ValueError(f"Rules YAML root must be a mapping, got {type(raw).__name__}")

    detection = _parse_detection_rule(raw.get("detection") or {})

    zones_raw = raw.get("zones") or []
    if not isinstance(zones_raw, list):
        raise ValueError(f"zones must be a list, got {type(zones_raw).__name__}")
    zones = [_parse_zone(z, i) for i, z in enumerate(zones_raw)]

    motion_gate = _parse_motion_gate(raw.get("motion_gate") or {})

    return Rules(detection=detection, zones=zones, motion_gate=motion_gate)
