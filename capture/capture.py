"""영상 수신 모듈. source_type만 바꾸면 소스 교체."""

import base64
import io
import logging
import os
import queue
import re
import threading
import time
from datetime import datetime, timezone

# RTSP는 기본 UDP라 패킷 손실이 많고, 끊김 시 소켓이 무한 블로킹됨.
# OpenCV가 cv2를 import하기 전에 환경변수를 읽으므로 여기서 설정.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000",
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from config import CaptureConfig  # noqa: E402

logger = logging.getLogger("capture")


def _mask_rtsp_credentials(uri: str) -> str:
    """rtsp://id:pw@host → rtsp://***@host 로 마스킹."""
    return re.sub(r"(rtsp://)([^/@]+)@", r"\1***@", uri)


class FrameCapture:
    """영상 소스에서 프레임을 캡처하여 큐에 넣는 워커."""

    # RTSP freeze 감지 임계값 (초). 이 시간 동안 새 프레임이 없으면 강제 재연결.
    RTSP_STALL_THRESHOLD_SEC = 10.0

    def __init__(self, config: CaptureConfig, output_queue: queue.Queue):
        self._config = config
        self._queue = output_queue
        self._running = False
        self._thread = None
        self._frame_count = 0
        self._dropped = 0  # 큐 full 상태에서 버려진 프레임 수
        self._source: VideoSource | None = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info(f"Capture 시작 [type={self._config.source_type}, uri={self._config.source_uri}]")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(
            f"Capture 종료 (총 {self._frame_count}프레임, 드롭: {self._dropped})"
        )

    def _capture_loop(self):
        self._source = self._create_source()

        try:
            while self._running:
                # RTSP freeze 워치독: 장시간 프레임 없으면 강제 재연결
                if isinstance(self._source, RTSPSource) and \
                        self._source.is_stalled(self.RTSP_STALL_THRESHOLD_SEC):
                    self._source.force_reconnect()

                frame = self._source.read()
                if frame is None:
                    logger.warning("프레임 읽기 실패, 재시도...")
                    time.sleep(0.5)
                    continue

                frame_data = self._encode_frame(frame)

                # 큐가 가득 차면 오래된 프레임 드롭 (실시간 특성)
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                        self._dropped += 1
                    except queue.Empty:
                        pass

                self._queue.put(frame_data)

                if self._frame_count % 10 == 0:
                    logger.info(f"캡처 중... frame_id={self._frame_count} fps={self._config.fps}")

                # fps는 매 루프에서 재평가 (런타임 변경 반영)
                fps = max(self._config.fps, 1)
                time.sleep(1.0 / fps)
        finally:
            self._source.release()

    def status(self) -> dict:
        """현재 캡처 상태 스냅샷. 웹UI/관리 API에서 조회."""
        src = self._source
        base = {
            "source_type": self._config.source_type,
            "frame_count": self._frame_count,
            "dropped": self._dropped,
            "fps_target": self._config.fps,
            "running": self._running,
        }
        if isinstance(src, RTSPSource):
            base.update({
                "rtsp_uri": src._masked_uri,
                "connected": src._cap is not None,
                "reconnect_count": src.reconnect_count,
                "last_frame_ts": src.last_frame_ts,
                "seconds_since_last_frame": (
                    time.time() - src.last_frame_ts
                    if src.last_frame_ts > 0 else None
                ),
                "stalled": src.is_stalled(self.RTSP_STALL_THRESHOLD_SEC),
                "stall_threshold_sec": self.RTSP_STALL_THRESHOLD_SEC,
            })
        return base

    def _create_source(self) -> "VideoSource":
        """source_type에 따라 소스 생성. 새 소스를 추가하려면 여기에 분기 추가."""
        source_type = self._config.source_type
        if source_type == "webcam":
            return WebcamSource(int(self._config.source_uri), self._config)
        elif source_type == "file":
            return FileSource(self._config.source_uri, self._config)
        elif source_type == "rtsp":
            return RTSPSource(self._config.source_uri, self._config)
        elif source_type == "synthetic":
            return SyntheticSource(self._config)
        else:
            raise ValueError(f"지원하지 않는 source_type: {source_type}")

    def _encode_frame(self, frame: np.ndarray) -> dict:
        """프레임을 API 전송 가능한 형태로 인코딩."""
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        self._frame_count += 1
        return {
            "frame_id": self._frame_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_base64": image_b64,
        }


# --- 영상 소스 구현체 (교체 가능) ---

class VideoSource:
    """영상 소스 인터페이스."""
    def read(self) -> np.ndarray | None:
        raise NotImplementedError

    def release(self):
        pass


class WebcamSource(VideoSource):
    def __init__(self, device_id: int, config: CaptureConfig):
        self._cap = cv2.VideoCapture(device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
        if not self._cap.isOpened():
            raise RuntimeError(f"웹캠 열기 실패: device_id={device_id}")
        logger.info(f"웹캠 소스 열림: device_id={device_id}")

    def read(self) -> np.ndarray | None:
        ret, frame = self._cap.read()
        return frame if ret else None

    def release(self):
        self._cap.release()


class FileSource(VideoSource):
    def __init__(self, path: str, config: CaptureConfig):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"파일 열기 실패: {path}")
        logger.info(f"파일 소스 열림: {path}")

    def read(self) -> np.ndarray | None:
        ret, frame = self._cap.read()
        if not ret:
            # 파일 끝이면 처음으로 되감기 (반복 재생)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
        return frame if ret else None

    def release(self):
        self._cap.release()


class RTSPSource(VideoSource):
    """RTSP 소스. 끊김 시 exponential backoff로 재연결하며 예외를 던지지 않는다.

    운영 중 CCTV 서버가 잠깐 꺼져도 파이프라인 전체가 죽지 않도록
    생성자에서 연결 실패해도 예외 대신 None 반환 상태로 시작한다.
    """

    INITIAL_BACKOFF_SEC = 1.0
    MAX_BACKOFF_SEC = 30.0

    def __init__(self, uri: str, config: CaptureConfig):
        self._uri = uri
        self._masked_uri = _mask_rtsp_credentials(uri)
        self._cap: cv2.VideoCapture | None = None
        self._backoff = self.INITIAL_BACKOFF_SEC
        self._last_attempt = 0.0
        self.reconnect_count = 0
        self.last_frame_ts: float = 0.0
        self._open()

    def _open(self) -> bool:
        """연결 시도. 성공 시 True, 실패 시 False (예외 없음)."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._last_attempt = time.time()

        cap = cv2.VideoCapture(self._uri, cv2.CAP_FFMPEG)
        # 버퍼 누적으로 인한 지연 방지 — 항상 최신 프레임만
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            logger.warning(
                f"RTSP 연결 실패 [{self._masked_uri}], "
                f"{self._backoff:.1f}s 후 재시도"
            )
            return False

        self._cap = cap
        self._backoff = self.INITIAL_BACKOFF_SEC  # 성공 시 backoff 리셋
        logger.info(f"RTSP 소스 연결: {self._masked_uri}")
        return True

    def _schedule_reconnect(self):
        """재연결 폭주 방지. backoff를 두 배로 늘리며 상한에서 캡."""
        self._backoff = min(self._backoff * 2, self.MAX_BACKOFF_SEC)

    def read(self) -> np.ndarray | None:
        # 끊긴 상태: backoff 경과 후에만 재시도
        if self._cap is None:
            if time.time() - self._last_attempt >= self._backoff:
                self.reconnect_count += 1
                if not self._open():
                    self._schedule_reconnect()
            return None

        ret, frame = self._cap.read()
        if not ret:
            logger.warning(
                f"RTSP 프레임 실패 [{self._masked_uri}], backoff={self._backoff:.1f}s"
            )
            self._cap.release()
            self._cap = None
            self._last_attempt = time.time()
            return None

        self.last_frame_ts = time.time()
        return frame

    def is_stalled(self, threshold_sec: float) -> bool:
        """마지막 프레임 이후 threshold 초 이상 경과했으면 True.

        RTSP는 ret=True로 오래된 프레임만 계속 오는 freeze 상태가 있어서
        재연결 트리거 여부를 외부(FrameCapture 워치독)에서 결정하도록 노출.
        last_frame_ts=0 (아직 한 번도 성공한 적 없음)인 경우는 False.
        """
        if self.last_frame_ts == 0.0:
            return False
        return (time.time() - self.last_frame_ts) >= threshold_sec

    def force_reconnect(self) -> None:
        """워치독이 호출. 다음 read()에서 backoff 대기 없이 즉시 재연결."""
        logger.warning(f"RTSP 강제 재연결 [{self._masked_uri}]")
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._last_attempt = 0.0
        self._backoff = self.INITIAL_BACKOFF_SEC

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class SyntheticSource(VideoSource):
    """테스트용 합성 프레임 생성. 외부 의존성 없음."""
    def __init__(self, config: CaptureConfig):
        self._w = config.frame_width
        self._h = config.frame_height
        self._count = 0
        logger.info(f"합성 소스 생성: {self._w}x{self._h}")

    def read(self) -> np.ndarray | None:
        frame = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        # 움직이는 사각형으로 시각적 변화
        x = (self._count * 5) % self._w
        cv2.rectangle(frame, (x, 100), (x + 80, 250), (0, 255, 0), -1)
        cv2.putText(frame, f"Frame {self._count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, ts, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        self._count += 1
        return frame
