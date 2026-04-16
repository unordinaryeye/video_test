"""RTSPSource 재연결 로직 단위 테스트.

실제 RTSP 서버 없이 cv2.VideoCapture를 모킹해서 backoff/재연결/마스킹 검증.
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from capture.capture import RTSPSource, _mask_rtsp_credentials
from config import CaptureConfig


def _make_cap(opened: bool, frames: list | None = None) -> MagicMock:
    """VideoCapture 모의 객체. frames를 순서대로 반환."""
    cap = MagicMock()
    cap.isOpened.return_value = opened
    seq = frames if frames is not None else []

    def read():
        if not seq:
            return (False, None)
        item = seq.pop(0)
        if item is None:
            return (False, None)
        return (True, item)

    cap.read.side_effect = read
    return cap


def test_mask_rtsp_credentials_hides_password():
    masked = _mask_rtsp_credentials("rtsp://admin:secret123@192.168.0.1:554/stream")
    assert "secret123" not in masked
    assert "admin" not in masked
    assert "192.168.0.1:554/stream" in masked


def test_mask_rtsp_credentials_passes_uri_without_auth():
    uri = "rtsp://192.168.0.1:554/stream"
    assert _mask_rtsp_credentials(uri) == uri


def test_constructor_does_not_raise_on_connection_failure():
    """생성 시 서버가 꺼져있어도 예외 없이 객체 생성되어야 한다."""
    with patch("capture.capture.cv2.VideoCapture", return_value=_make_cap(opened=False)):
        source = RTSPSource("rtsp://localhost:554/x", CaptureConfig())
        assert source._cap is None
        assert source.read() is None  # 끊긴 상태에서도 예외 없이 None


def test_successful_read_returns_frame_and_updates_timestamp():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    cap = _make_cap(opened=True, frames=[frame])
    with patch("capture.capture.cv2.VideoCapture", return_value=cap):
        source = RTSPSource("rtsp://localhost/x", CaptureConfig())
        result = source.read()
        assert result is not None
        assert source.last_frame_ts > 0


def test_backoff_prevents_reconnect_spam():
    """끊긴 상태에서 read()를 연속 호출해도 backoff 전에는 재연결 시도 안 함."""
    cap_closed = _make_cap(opened=False)
    with patch("capture.capture.cv2.VideoCapture", return_value=cap_closed) as mock_cv:
        source = RTSPSource("rtsp://localhost/x", CaptureConfig())
        initial_calls = mock_cv.call_count
        for _ in range(100):
            source.read()
        # backoff(1초) 이내에는 추가 연결 시도가 거의 없어야 한다
        assert mock_cv.call_count - initial_calls <= 1


def test_backoff_doubles_on_repeated_failure():
    source = RTSPSource.__new__(RTSPSource)
    source._backoff = RTSPSource.INITIAL_BACKOFF_SEC
    source._schedule_reconnect()
    assert source._backoff == 2.0
    source._schedule_reconnect()
    assert source._backoff == 4.0


def test_backoff_capped_at_max():
    source = RTSPSource.__new__(RTSPSource)
    source._backoff = RTSPSource.MAX_BACKOFF_SEC
    source._schedule_reconnect()
    assert source._backoff == RTSPSource.MAX_BACKOFF_SEC


def test_backoff_resets_on_successful_open():
    """재연결 성공하면 backoff가 초기값으로 리셋되어야 한다."""
    cap_open = _make_cap(opened=True, frames=[np.zeros((1, 1, 3), dtype=np.uint8)])
    with patch("capture.capture.cv2.VideoCapture", return_value=cap_open):
        source = RTSPSource("rtsp://localhost/x", CaptureConfig())
        source._backoff = 16.0  # 이전에 실패 누적됐다고 가정
        source._open()
        assert source._backoff == RTSPSource.INITIAL_BACKOFF_SEC


def test_read_failure_marks_disconnected_and_increments_count():
    """read() 실패 시 _cap을 None으로 만들어 다음 호출에서 재연결 트리거."""
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    cap_failing = _make_cap(opened=True, frames=[frame, None])  # 1회 성공 후 실패

    with patch("capture.capture.cv2.VideoCapture", return_value=cap_failing):
        source = RTSPSource("rtsp://localhost/x", CaptureConfig())
        assert source.read() is not None
        assert source.read() is None  # 실패
        assert source._cap is None
        initial_count = source.reconnect_count

        # backoff 없이 즉시 재시도할 수 있도록 last_attempt 과거로
        source._last_attempt = 0.0
        source.read()
        assert source.reconnect_count == initial_count + 1


def test_release_is_idempotent():
    """release를 여러 번 호출해도 에러 없어야 한다."""
    with patch("capture.capture.cv2.VideoCapture", return_value=_make_cap(opened=False)):
        source = RTSPSource("rtsp://localhost/x", CaptureConfig())
        source.release()
        source.release()  # 두 번 호출해도 안전
