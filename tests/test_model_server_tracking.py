"""model_server 트래킹 통합 테스트 (FastAPI TestClient + mock 사용)."""

import base64
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _make_base64_image(width: int = 64, height: int = 64) -> str:
    """테스트용 더미 이미지를 base64로 인코딩."""
    img = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _make_mock_box(cls_id: int = 0, conf: float = 0.9, track_id: int | None = None):
    """YOLO Box 목 객체 생성."""
    box = MagicMock()
    box.cls = [cls_id]
    box.conf = [conf]
    box.xyxy = [[10.0, 20.0, 100.0, 200.0]]
    box.id = [track_id] if track_id is not None else None
    return box


def _make_mock_result(names: dict, boxes: list):
    """YOLO Result 목 객체 생성."""
    result = MagicMock()
    result.names = names
    result.boxes = boxes
    return result


COCO_NAMES = {0: "person", 1: "bicycle", 2: "car"}


@pytest.fixture()
def mock_model():
    """YOLO 모델을 목으로 대체하는 픽스처."""
    model = MagicMock()
    model.names = COCO_NAMES

    # 기본 detection 결과 (use_tracking=False)
    model.return_value = [
        _make_mock_result(COCO_NAMES, [_make_mock_box(cls_id=0, conf=0.85)])
    ]
    # track() 결과 (use_tracking=True)
    model.track.return_value = [
        _make_mock_result(COCO_NAMES, [_make_mock_box(cls_id=0, conf=0.85, track_id=1)])
    ]
    return model


@pytest.fixture()
def client(mock_model):
    """목 모델이 주입된 TestClient (startup의 load_model도 목으로 치환)."""
    from model_server import app as app_module

    def _fake_load_model():
        app_module._model = mock_model
        return mock_model

    with patch.object(app_module, "_model", mock_model), \
         patch.object(app_module, "load_model", _fake_load_model):
        with TestClient(app_module.app, raise_server_exceptions=True) as c:
            yield c


# ── 테스트 케이스 ──────────────────────────────────────────────

class TestClassesEndpoint:
    def test_classes_contains_person(self, client):
        resp = client.get("/classes")
        assert resp.status_code == 200
        data = resp.json()
        assert "classes" in data
        assert "person" in data["classes"]

    def test_classes_returns_list(self, client):
        resp = client.get("/classes")
        assert isinstance(resp.json()["classes"], list)


class TestPredictNoTracking:
    def test_track_id_is_none_when_tracking_disabled(self, client):
        payload = {
            "image_base64": _make_base64_image(),
            "frame_id": 1,
            "use_tracking": False,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "detections" in data
        for det in data["detections"]:
            assert det["track_id"] is None, f"track_id should be None, got {det['track_id']}"

    def test_response_schema_valid(self, client):
        payload = {
            "image_base64": _make_base64_image(),
            "frame_id": 42,
            "use_tracking": False,
        }
        resp = client.post("/predict", json=payload)
        data = resp.json()
        assert data["frame_id"] == 42
        assert isinstance(data["inference_time_ms"], float)
        assert isinstance(data["detections"], list)


class TestPredictWithTracking:
    def test_schema_valid_with_tracking(self, client):
        """use_tracking=True 호출 시 응답 스키마가 유효한지 검증."""
        payload = {
            "image_base64": _make_base64_image(),
            "frame_id": 10,
            "use_tracking": True,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "detections" in data
        assert "frame_id" in data
        assert "inference_time_ms" in data

    def test_track_id_assigned_when_tracking_enabled(self, client):
        """use_tracking=True 시 track_id가 정수로 채워지는지 확인."""
        payload = {
            "image_base64": _make_base64_image(),
            "frame_id": 10,
            "use_tracking": True,
        }
        resp = client.post("/predict", json=payload)
        data = resp.json()
        assert len(data["detections"]) > 0
        for det in data["detections"]:
            assert isinstance(det["track_id"], int), (
                f"track_id should be int when tracking enabled, got {det['track_id']}"
            )

    def test_model_track_called_not_call(self, mock_model, client):
        """use_tracking=True 시 model.track()이 호출되고 model()은 호출되지 않는지 확인."""
        payload = {
            "image_base64": _make_base64_image(),
            "frame_id": 5,
            "use_tracking": True,
        }
        client.post("/predict", json=payload)
        mock_model.track.assert_called_once()
        mock_model.assert_not_called()
