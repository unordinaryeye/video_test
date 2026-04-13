"""tests/test_rules_loader.py — rules/loader.py 단위 테스트."""

import os
import textwrap

import pytest

from rules.loader import (
    DetectionRule,
    MotionGate,
    Rules,
    Zone,
    ZoneRule,
    load_rules,
)


# ---------------------------------------------------------------------------
# load_rules — 파일 없음 / None
# ---------------------------------------------------------------------------

def test_load_rules_none_returns_defaults():
    rules = load_rules(None)
    assert isinstance(rules, Rules)
    assert rules.detection.classes == []
    assert rules.detection.min_confidence == 0.0
    assert rules.zones == []
    assert rules.motion_gate.enabled is False


def test_load_rules_missing_file_returns_defaults(tmp_path):
    rules = load_rules(str(tmp_path / "nonexistent.yaml"))
    assert isinstance(rules, Rules)
    assert rules.zones == []


# ---------------------------------------------------------------------------
# load_rules — 정상 파일
# ---------------------------------------------------------------------------

def test_load_rules_full_yaml(tmp_path):
    yaml_content = textwrap.dedent("""
        detection:
          classes: [person, hardhat]
          min_confidence: 0.6

        zones:
          - name: "위험구역_A"
            polygon: [[0,0],[100,0],[100,100],[0,100]]
            rules:
              - type: entry
                classes: [person]
              - type: count_exceeds
                threshold: 3
                classes: [person]

        motion_gate:
          enabled: false
          idle_seconds: 120
          movement_threshold_px: 15
          classes: [person]
    """)
    p = tmp_path / "rules.yaml"
    p.write_text(yaml_content, encoding="utf-8")

    rules = load_rules(str(p))

    assert rules.detection.classes == ["person", "hardhat"]
    assert rules.detection.min_confidence == pytest.approx(0.6)
    assert len(rules.zones) == 1
    assert rules.zones[0].name == "위험구역_A"
    assert len(rules.zones[0].rules) == 2
    assert rules.zones[0].rules[1].threshold == 3
    assert rules.motion_gate.idle_seconds == pytest.approx(120.0)


def test_load_rules_empty_yaml(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    rules = load_rules(str(p))
    assert isinstance(rules, Rules)


def test_load_rules_detection_only(tmp_path):
    yaml_content = "detection:\n  min_confidence: 0.3\n"
    p = tmp_path / "r.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    rules = load_rules(str(p))
    assert rules.detection.min_confidence == pytest.approx(0.3)
    assert rules.zones == []


# ---------------------------------------------------------------------------
# 스키마 검증 — 에러 케이스
# ---------------------------------------------------------------------------

def test_invalid_confidence_out_of_range(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("detection:\n  min_confidence: 1.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="min_confidence"):
        load_rules(str(p))


def test_invalid_zone_missing_name(tmp_path):
    yaml_content = textwrap.dedent("""
        zones:
          - polygon: [[0,0],[1,0],[1,1]]
            rules: []
    """)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(ValueError, match="name"):
        load_rules(str(p))


def test_invalid_zone_polygon_too_small(tmp_path):
    yaml_content = textwrap.dedent("""
        zones:
          - name: "Z"
            polygon: [[0,0],[1,0]]
            rules: []
    """)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(ValueError, match="polygon"):
        load_rules(str(p))


def test_invalid_zone_rule_type(tmp_path):
    yaml_content = textwrap.dedent("""
        zones:
          - name: "Z"
            polygon: [[0,0],[1,0],[1,1]]
            rules:
              - type: unknown_type
    """)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(ValueError, match="type"):
        load_rules(str(p))


def test_invalid_motion_gate_idle_seconds(tmp_path):
    yaml_content = "motion_gate:\n  idle_seconds: -1\n"
    p = tmp_path / "bad.yaml"
    p.write_text(yaml_content, encoding="utf-8")
    with pytest.raises(ValueError, match="idle_seconds"):
        load_rules(str(p))
