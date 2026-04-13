"""rules/state.py — RulesState 단위 테스트."""

import pytest

from rules.state import RulesState


@pytest.fixture()
def yaml_path(tmp_path):
    p = tmp_path / "rules.yaml"
    p.write_text(
        "detection:\n  classes: [person]\n  min_confidence: 0.5\n",
        encoding="utf-8",
    )
    return str(p)


def test_init_without_path_uses_defaults():
    state = RulesState(None)
    rules, _ = state.current()
    assert rules.detection.classes == []
    assert rules.detection.min_confidence == 0.0


def test_init_with_path_loads_yaml(yaml_path):
    state = RulesState(yaml_path)
    rules, _ = state.current()
    assert rules.detection.classes == ["person"]
    assert rules.detection.min_confidence == 0.5


def test_read_yaml_text_returns_file_content(yaml_path):
    state = RulesState(yaml_path)
    text = state.read_yaml_text()
    assert "person" in text


def test_read_yaml_text_empty_when_no_path():
    assert RulesState(None).read_yaml_text() == ""


def test_update_from_yaml_applies_and_persists(yaml_path):
    state = RulesState(yaml_path)
    new_yaml = "detection:\n  classes: [car]\n  min_confidence: 0.8\n"
    state.update_from_yaml(new_yaml)
    rules, _ = state.current()
    assert rules.detection.classes == ["car"]
    assert rules.detection.min_confidence == 0.8
    # 파일에도 저장되었는지
    with open(yaml_path, "r", encoding="utf-8") as f:
        assert "car" in f.read()


def test_update_from_yaml_rejects_invalid():
    state = RulesState(None)
    with pytest.raises(ValueError):
        state.update_from_yaml("detection:\n  min_confidence: 2.0\n")


def test_reload_from_file_picks_up_external_edit(yaml_path):
    state = RulesState(yaml_path)
    # 외부에서 파일 수정
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("detection:\n  classes: [truck]\n  min_confidence: 0.3\n")
    state.reload_from_file()
    rules, _ = state.current()
    assert rules.detection.classes == ["truck"]


def test_current_returns_motion_gate_state():
    state = RulesState(None)
    _, mg = state.current()
    # idle_ids should be [] for no-op default
    assert mg.update([], None) == []
