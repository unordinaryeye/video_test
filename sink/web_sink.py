"""웹 뷰어 Sink. FastAPI로 MJPEG 스트림과 HTML 페이지를 제공."""

import base64
import logging
import threading
import time
from collections import deque

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

logger = logging.getLogger("sink.web")


HTML_PAGE = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>실시간 영상 추론 파이프라인</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           background: #0f172a; color: #e2e8f0; }
    header { padding: 16px 24px; background: #1e293b; border-bottom: 1px solid #334155; }
    header h1 { margin: 0; font-size: 18px; font-weight: 600; }
    header p { margin: 4px 0 0; font-size: 13px; color: #94a3b8; }
    .container { display: flex; gap: 16px; padding: 16px; max-width: 1400px; margin: 0 auto; }
    .video-box { flex: 2; background: #1e293b; border-radius: 8px; padding: 16px;
                 border: 1px solid #334155; }
    .video-box h2 { margin: 0 0 12px; font-size: 14px; color: #94a3b8; font-weight: 500; }
    .video-box img { width: 100%; border-radius: 6px; background: #0f172a; }
    .sidebar { flex: 1; display: flex; flex-direction: column; gap: 16px; }
    .card { background: #1e293b; border-radius: 8px; padding: 16px;
            border: 1px solid #334155; }
    .card h2 { margin: 0 0 12px; font-size: 14px; color: #94a3b8; font-weight: 500; }
    .stat { display: flex; justify-content: space-between; margin: 8px 0;
            padding: 8px 0; border-bottom: 1px solid #334155; font-size: 14px; }
    .stat:last-child { border-bottom: none; }
    .stat-label { color: #94a3b8; }
    .stat-value { color: #22d3ee; font-weight: 600; font-family: ui-monospace, monospace; }
    .log { font-family: ui-monospace, monospace; font-size: 12px;
           max-height: 400px; overflow-y: auto; background: #0f172a;
           padding: 12px; border-radius: 6px; border: 1px solid #334155; }
    .log-entry { margin: 4px 0; color: #cbd5e1; }
    .pipeline { display: flex; align-items: center; justify-content: space-between;
                margin-top: 8px; }
    .stage { flex: 1; text-align: center; padding: 12px 8px; background: #0f172a;
             border-radius: 6px; border: 1px solid #334155; }
    .stage-name { font-size: 12px; color: #94a3b8; }
    .stage-status { font-size: 16px; margin-top: 4px; }
    .arrow { color: #475569; margin: 0 8px; }
    .active { border-color: #22d3ee; }
    .active .stage-status { color: #22d3ee; }
  </style>
</head>
<body>
  <header>
    <h1>실시간 영상 추론 파이프라인</h1>
    <p>Capture → Inference (YOLOv8) → Sink | 프로토타입 데모</p>
  </header>

  <div class="container">
    <div class="video-box">
      <h2>실시간 영상 + 감지 결과</h2>
      <img src="/stream" alt="video stream" />
    </div>

    <div class="sidebar">
      <div class="card">
        <h2>파이프라인 상태</h2>
        <div class="pipeline">
          <div class="stage active">
            <div class="stage-name">Capture</div>
            <div class="stage-status">●</div>
          </div>
          <span class="arrow">→</span>
          <div class="stage active">
            <div class="stage-name">Inference</div>
            <div class="stage-status">●</div>
          </div>
          <span class="arrow">→</span>
          <div class="stage active">
            <div class="stage-name">Sink</div>
            <div class="stage-status">●</div>
          </div>
        </div>
      </div>

      <div class="card">
        <h2>통계</h2>
        <div class="stat">
          <span class="stat-label">처리 프레임</span>
          <span class="stat-value" id="frame-count">0</span>
        </div>
        <div class="stat">
          <span class="stat-label">최근 감지 수</span>
          <span class="stat-value" id="detect-count">0</span>
        </div>
        <div class="stat">
          <span class="stat-label">추론 지연 (ms)</span>
          <span class="stat-value" id="latency">0</span>
        </div>
        <div class="stat">
          <span class="stat-label">FPS</span>
          <span class="stat-value" id="fps">0</span>
        </div>
      </div>

      <div class="card">
        <h2>최근 감지 로그</h2>
        <div class="log" id="log"></div>
      </div>
    </div>
  </div>

  <script>
    async function poll() {
      try {
        const res = await fetch("/stats");
        const data = await res.json();
        document.getElementById("frame-count").textContent = data.frame_count;
        document.getElementById("detect-count").textContent = data.last_detections;
        document.getElementById("latency").textContent = data.last_latency_ms;
        document.getElementById("fps").textContent = data.fps.toFixed(1);

        const log = document.getElementById("log");
        log.innerHTML = data.log.map(e =>
          '<div class="log-entry">' + e + '</div>'
        ).join("");
      } catch (e) { console.error(e); }
    }
    setInterval(poll, 500);
    poll();
  </script>
</body>
</html>
"""


class WebSink:
    """FastAPI 기반 실시간 뷰어. 프레임에 감지 박스를 그려 MJPEG로 스트리밍."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        self._latest_jpeg: bytes | None = None
        self._lock = threading.Lock()

        # 통계
        self._frame_count = 0
        self._last_detections = 0
        self._last_latency = 0.0
        self._log: deque[str] = deque(maxlen=20)
        self._start_time = time.time()

        self._app = self._build_app()
        self._server_thread = threading.Thread(
            target=self._run_server,
            args=(host, port),
            daemon=True,
        )
        self._server_thread.start()
        logger.info(f"WebSink 서버 시작: http://{host}:{port}")

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="Pipeline Viewer")

        @app.get("/", response_class=HTMLResponse)
        def index():
            return HTML_PAGE

        @app.get("/stream")
        def stream():
            return StreamingResponse(
                self._mjpeg_generator(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        @app.get("/stats")
        def stats():
            elapsed = max(time.time() - self._start_time, 0.001)
            return {
                "frame_count": self._frame_count,
                "last_detections": self._last_detections,
                "last_latency_ms": round(self._last_latency, 1),
                "fps": self._frame_count / elapsed,
                "log": list(self._log),
            }

        return app

    def _run_server(self, host: str, port: int):
        uvicorn.run(self._app, host=host, port=port, log_level="warning")

    def _mjpeg_generator(self):
        while True:
            with self._lock:
                jpeg = self._latest_jpeg
            if jpeg is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                )
            time.sleep(0.05)  # ~20fps 상한

    def send(self, result: dict) -> None:
        """추론 결과를 받아 박스를 그리고 JPEG로 갱신."""
        image_b64 = result.get("image_base64")
        if not image_b64:
            return

        # Base64 → numpy
        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return

        detections = result.get("detections", [])
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            label = f"{det['label']} {det['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (34, 211, 238), 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(label) * 9, y1), (34, 211, 238), -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (15, 23, 42), 1)

        # 좌상단 오버레이
        frame_id = result.get("frame_id", 0)
        latency = result.get("inference_time_ms", 0)
        overlay = f"Frame #{frame_id} | {latency:.0f}ms | {len(detections)} det"
        cv2.putText(frame, overlay, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 211, 238), 2)

        # JPEG 인코딩
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return

        with self._lock:
            self._latest_jpeg = buffer.tobytes()
            self._frame_count += 1
            self._last_detections = len(detections)
            self._last_latency = latency
            labels = ", ".join(d["label"] for d in detections[:3]) or "(감지 없음)"
            self._log.appendleft(f"#{frame_id} [{latency:.0f}ms] {labels}")

    def close(self) -> None:
        # uvicorn은 daemon thread라 프로세스 종료 시 자동 정리
        logger.info(f"WebSink 종료 (처리: {self._frame_count}프레임)")
