"""룰 런타임 상태. YAML 재로드와 스레드 안전한 공유 접근을 제공."""

from __future__ import annotations

import threading
from typing import Optional

from rules.loader import Rules, load_rules
from rules.motion_gate import MotionGateState


class RulesState:
    """파이프라인이 공유하는 룰 상태.

    InferenceWorker는 매 프레임 current() 로 최신 Rules/MotionGateState를 읽고,
    WebSink 관리 엔드포인트는 reload_from_file() 또는 update_from_yaml() 로 교체한다.
    """

    def __init__(self, path: Optional[str] = None):
        self._lock = threading.Lock()
        self._path = path
        self._rules: Rules = load_rules(path)
        self._motion_gate = MotionGateState(self._rules.motion_gate)

    @property
    def path(self) -> Optional[str]:
        return self._path

    def current(self) -> tuple[Rules, MotionGateState]:
        with self._lock:
            return self._rules, self._motion_gate

    def reload_from_file(self) -> Rules:
        """현재 path의 YAML을 다시 읽어 rules/motion_gate를 교체."""
        new_rules = load_rules(self._path)
        with self._lock:
            self._rules = new_rules
            self._motion_gate = MotionGateState(new_rules.motion_gate)
        return new_rules

    def update_from_yaml(self, yaml_text: str) -> Rules:
        """YAML 문자열을 파싱해 적용. 파일에 저장도 수행 (path가 있을 때).

        Raises:
            ValueError: 스키마 검증 실패 시.
        """
        import yaml
        from rules.loader import (
            _parse_detection_rule,
            _parse_motion_gate,
            _parse_zone,
        )

        try:
            raw = yaml.safe_load(yaml_text)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML parse error: {e}") from e

        if raw is None:
            new_rules = Rules()
        elif not isinstance(raw, dict):
            raise ValueError("YAML root must be a mapping")
        else:
            detection = _parse_detection_rule(raw.get("detection") or {})
            zones_raw = raw.get("zones") or []
            if not isinstance(zones_raw, list):
                raise ValueError("zones must be a list")
            zones = [_parse_zone(z, i) for i, z in enumerate(zones_raw)]
            motion_gate = _parse_motion_gate(raw.get("motion_gate") or {})
            new_rules = Rules(detection=detection, zones=zones, motion_gate=motion_gate)

        with self._lock:
            self._rules = new_rules
            self._motion_gate = MotionGateState(new_rules.motion_gate)

        if self._path:
            with open(self._path, "w", encoding="utf-8") as f:
                f.write(yaml_text)

        return new_rules

    def read_yaml_text(self) -> str:
        """현재 YAML 파일 원문을 읽어 반환. path 없거나 파일 없으면 빈 문자열."""
        import os
        if not self._path or not os.path.exists(self._path):
            return ""
        with open(self._path, "r", encoding="utf-8") as f:
            return f.read()
