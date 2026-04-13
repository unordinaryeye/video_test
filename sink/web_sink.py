"""웹 뷰어 Sink. FastAPI로 MJPEG 스트림, HTML 페이지, 관리 엔드포인트 제공."""

import base64
import logging
import os
import threading
import time
from collections import deque
from typing import Optional

import cv2
import httpx
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

logger = logging.getLogger("sink.web")


_FONT_CANDIDATES = [
    "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def _load_font(size: int = 14):
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


_FONT = _load_font(14)
_FONT_SMALL = _load_font(12)


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
    .container { display: flex; gap: 16px; padding: 16px; max-width: 1600px; margin: 0 auto; }
    .video-box { flex: 2; background: #1e293b; border-radius: 8px; padding: 16px;
                 border: 1px solid #334155; }
    .video-box h2 { margin: 0 0 12px; font-size: 14px; color: #94a3b8; font-weight: 500; }
    .video-box img { width: 100%; border-radius: 6px; background: #0f172a; }
    .sidebar { flex: 1; display: flex; flex-direction: column; gap: 16px; min-width: 360px; }
    .card { background: #1e293b; border-radius: 8px; padding: 16px;
            border: 1px solid #334155; }
    .card h2 { margin: 0 0 12px; font-size: 14px; color: #94a3b8; font-weight: 500;
               display: flex; justify-content: space-between; align-items: center; }
    .stat { display: flex; justify-content: space-between; margin: 8px 0;
            padding: 8px 0; border-bottom: 1px solid #334155; font-size: 14px; }
    .stat:last-child { border-bottom: none; }
    .stat-label { color: #94a3b8; }
    .stat-value { color: #22d3ee; font-weight: 600; font-family: ui-monospace, monospace; }
    .log { font-family: ui-monospace, monospace; font-size: 12px;
           max-height: 240px; overflow-y: auto; background: #0f172a;
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
    textarea { width: 100%; min-height: 240px; box-sizing: border-box; background: #0f172a;
               color: #e2e8f0; border: 1px solid #334155; border-radius: 6px;
               padding: 10px; font-family: ui-monospace, monospace; font-size: 12px;
               resize: vertical; }
    button { background: #22d3ee; color: #0f172a; border: 0; padding: 8px 14px;
             border-radius: 6px; font-weight: 600; cursor: pointer; font-size: 13px; }
    button:hover { background: #67e8f9; }
    button.secondary { background: #334155; color: #e2e8f0; }
    button.secondary:hover { background: #475569; }
    .row { display: flex; gap: 8px; align-items: center; margin-top: 10px; flex-wrap: wrap; }
    .toggle { display: inline-flex; gap: 8px; align-items: center; cursor: pointer;
              user-select: none; font-size: 13px; }
    .toggle input { width: 16px; height: 16px; cursor: pointer; }
    .msg { font-size: 12px; margin-top: 8px; min-height: 16px; }
    .msg.ok { color: #22c55e; }
    .msg.err { color: #ef4444; }
    details { margin-top: 10px; }
    details summary { cursor: pointer; font-size: 12px; color: #94a3b8; }
    .classes { font-family: ui-monospace, monospace; font-size: 11px; color: #94a3b8;
               max-height: 120px; overflow-y: auto; background: #0f172a; padding: 8px;
               border-radius: 6px; border: 1px solid #334155; margin-top: 6px; }
    .disabled-card { opacity: 0.5; pointer-events: none; }
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
      <div style="margin-top:8px; font-size:12px; color:#94a3b8; display:flex; gap:16px; flex-wrap:wrap;">
        <span><span style="display:inline-block;width:10px;height:10px;background:#22d3ee;border-radius:2px;"></span> 통과 detection</span>
        <span><span style="display:inline-block;width:10px;height:10px;background:#ef4444;border-radius:2px;"></span> Zone 이벤트 발생</span>
        <span><span style="display:inline-block;width:10px;height:10px;border:2px solid #94a3b8;"></span> 비활성 Zone</span>
        <span><span style="display:inline-block;width:10px;height:10px;border:2px solid #ef4444;background:rgba(239,68,68,0.2);"></span> 활성 Zone</span>
      </div>
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
          <span class="stat-label">Zone 이벤트</span>
          <span class="stat-value" id="event-count">0</span>
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

      <div class="card" id="admin-card">
        <h2>관리 (실시간 반영)</h2>
        <label class="toggle">
          <input type="checkbox" id="tracking-toggle" />
          <span>객체 트래킹 (track_id 부여)</span>
        </label>
        <div class="row">
          <strong style="font-size:13px;">YAML 룰</strong>
          <button class="secondary" id="reload-btn" type="button">파일 리로드</button>
        </div>
        <textarea id="yaml-text" spellcheck="false" placeholder="--rules 플래그로 YAML 경로를 지정하세요"></textarea>
        <div class="row">
          <button id="save-btn" type="button">저장 + 적용</button>
          <span class="msg" id="save-msg"></span>
        </div>
        <details>
          <summary>지원 클래스 목록 (모델 서버)</summary>
          <div class="classes" id="classes">불러오는 중...</div>
        </details>
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
        document.getElementById("event-count").textContent = data.last_events;
        document.getElementById("latency").textContent = data.last_latency_ms;
        document.getElementById("fps").textContent = data.fps.toFixed(1);

        const log = document.getElementById("log");
        log.innerHTML = data.log.map(e =>
          '<div class="log-entry">' + e + '</div>'
        ).join("");
      } catch (e) { console.error(e); }
    }

    async function loadAdmin() {
      try {
        const t = await fetch("/tracking").then(r => r.json());
        document.getElementById("tracking-toggle").checked = !!t.enabled;
      } catch (e) {}
      try {
        const r = await fetch("/rules").then(r => r.json());
        document.getElementById("yaml-text").value = r.yaml || "";
      } catch (e) {}
      try {
        const c = await fetch("/classes").then(r => r.json());
        document.getElementById("classes").textContent = (c.classes || []).join(", ");
      } catch (e) {
        document.getElementById("classes").textContent = "(모델 서버 응답 없음)";
      }
    }

    document.getElementById("tracking-toggle").addEventListener("change", async (e) => {
      await fetch("/tracking", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({enabled: e.target.checked}),
      });
    });

    document.getElementById("save-btn").addEventListener("click", async () => {
      const msg = document.getElementById("save-msg");
      msg.textContent = "저장 중...";
      msg.className = "msg";
      const yaml = document.getElementById("yaml-text").value;
      try {
        const res = await fetch("/rules", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({yaml}),
        });
        if (res.ok) {
          msg.textContent = "✓ 적용됨";
          msg.className = "msg ok";
        } else {
          const err = await res.json();
          msg.textContent = "오류: " + (err.detail || res.status);
          msg.className = "msg err";
        }
      } catch (e) {
        msg.textContent = "오류: " + e.message;
        msg.className = "msg err";
      }
    });

    document.getElementById("reload-btn").addEventListener("click", async () => {
      await fetch("/rules/reload", {method: "POST"});
      const r = await fetch("/rules").then(r => r.json());
      document.getElementById("yaml-text").value = r.yaml || "";
    });

    setInterval(poll, 500);
    poll();
    loadAdmin();
  </script>
</body>
</html>
"""


class _RulesUpdate(BaseModel):
    yaml: str


class _TrackingUpdate(BaseModel):
    enabled: bool


class WebSink:
    """FastAPI 기반 실시간 뷰어 + 관리 API."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080,
                 admin_ctx: Optional[dict] = None):
        self._latest_jpeg: bytes | None = None
        self._lock = threading.Lock()

        self._frame_count = 0
        self._last_detections = 0
        self._last_events = 0
        self._last_latency = 0.0
        self._log: deque[str] = deque(maxlen=20)
        self._start_time = time.time()

        # admin_ctx: {"rules_state": RulesState, "model_config": ModelConfig}
        self._admin = admin_ctx or {}

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
                "last_events": self._last_events,
                "last_latency_ms": round(self._last_latency, 1),
                "fps": self._frame_count / elapsed,
                "log": list(self._log),
            }

        # --- 관리 엔드포인트 ---
        @app.get("/rules")
        def get_rules():
            rs = self._admin.get("rules_state")
            if rs is None:
                return {"path": None, "yaml": ""}
            return {"path": rs.path, "yaml": rs.read_yaml_text()}

        @app.post("/rules")
        def post_rules(body: _RulesUpdate):
            rs = self._admin.get("rules_state")
            if rs is None:
                raise HTTPException(503, "rules_state 미설정")
            try:
                rs.update_from_yaml(body.yaml)
            except ValueError as e:
                raise HTTPException(400, str(e))
            return {"ok": True}

        @app.post("/rules/reload")
        def reload_rules():
            rs = self._admin.get("rules_state")
            if rs is None:
                raise HTTPException(503, "rules_state 미설정")
            rs.reload_from_file()
            return {"ok": True}

        @app.get("/tracking")
        def get_tracking():
            mc = self._admin.get("model_config")
            return {"enabled": bool(mc.use_tracking) if mc else False}

        @app.post("/tracking")
        def post_tracking(body: _TrackingUpdate):
            mc = self._admin.get("model_config")
            if mc is None:
                raise HTTPException(503, "model_config 미설정")
            mc.use_tracking = bool(body.enabled)
            return {"enabled": mc.use_tracking}

        @app.get("/classes")
        def get_classes():
            mc = self._admin.get("model_config")
            if mc is None:
                return {"classes": []}
            url = mc.api_url.rsplit("/", 1)[0] + "/classes"
            try:
                with httpx.Client(timeout=2.0) as c:
                    r = c.get(url)
                    r.raise_for_status()
                    return r.json()
            except Exception as e:
                raise HTTPException(502, f"모델 서버 조회 실패: {e}")

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
            time.sleep(0.05)

    def _draw_overlays(self, frame: np.ndarray, detections: list, zone_events: list,
                       frame_id: int, latency: float) -> np.ndarray:
        """PIL로 zone polygon, detection box, 오버레이 텍스트를 그림."""
        rules_state = self._admin.get("rules_state")
        rules = rules_state.current()[0] if rules_state else None

        active_zones = {e["zone_name"] for e in zone_events}
        # 이벤트에 포함된 bbox 집합 (튜플로 키)
        event_bboxes = set()
        for e in zone_events:
            for d in e.get("detections", []):
                event_bboxes.add(tuple(d.get("bbox", [])))

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img, "RGBA")

        # 1) Zone 폴리곤
        if rules:
            for zone in rules.zones:
                pts = [(int(x), int(y)) for x, y in zone.polygon]
                if len(pts) < 3:
                    continue
                is_active = zone.name in active_zones
                outline = (239, 68, 68, 255) if is_active else (148, 163, 184, 255)
                fill = (239, 68, 68, 60) if is_active else (148, 163, 184, 30)
                draw.polygon(pts, fill=fill, outline=outline)
                # 이름
                lx, ly = pts[0]
                draw.rectangle([lx, ly - 20, lx + len(zone.name) * 10 + 10, ly],
                               fill=(15, 23, 42, 200))
                draw.text((lx + 4, ly - 18), zone.name, fill=outline, font=_FONT_SMALL)

        # 2) Detection 박스
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            in_event = tuple(bbox) in event_bboxes
            color = (239, 68, 68, 255) if in_event else (34, 211, 238, 255)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            tid = det.get("track_id")
            label = f"{det['label']} {det['confidence']:.2f}"
            if tid is not None:
                label = f"#{tid} " + label
            tw = len(label) * 8 + 8
            draw.rectangle([x1, y1 - 20, x1 + tw, y1], fill=color)
            draw.text((x1 + 4, y1 - 18), label, fill=(15, 23, 42), font=_FONT_SMALL)

        # 3) 좌상단 요약
        overlay = f"Frame #{frame_id} | {latency:.0f}ms | {len(detections)} det | {len(zone_events)} evt"
        draw.rectangle([0, 0, len(overlay) * 9 + 12, 28], fill=(15, 23, 42, 200))
        draw.text((8, 6), overlay, fill=(34, 211, 238), font=_FONT)

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def send(self, result: dict) -> None:
        image_b64 = result.get("image_base64")
        if not image_b64:
            return

        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if frame is None:
            return

        detections = result.get("detections", [])
        zone_events = result.get("zone_events", [])
        frame_id = result.get("frame_id", 0)
        latency = result.get("inference_time_ms", 0)

        frame = self._draw_overlays(frame, detections, zone_events, frame_id, latency)

        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return

        with self._lock:
            self._latest_jpeg = buffer.tobytes()
            self._frame_count += 1
            self._last_detections = len(detections)
            self._last_events = len(zone_events)
            self._last_latency = latency
            labels = ", ".join(d["label"] for d in detections[:3]) or "(감지 없음)"
            evt_suffix = f" | evt:{len(zone_events)}" if zone_events else ""
            self._log.appendleft(f"#{frame_id} [{latency:.0f}ms] {labels}{evt_suffix}")

    def close(self) -> None:
        logger.info(f"WebSink 종료 (처리: {self._frame_count}프레임)")
