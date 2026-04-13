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


_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def _load_html() -> str:
    with open(os.path.join(_STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
        return f.read()


def _empty_parsed() -> dict:
    return {
        "detection": {"classes": [], "min_confidence": 0.0},
        "zones": [],
        "motion_gate": {
            "enabled": False, "idle_seconds": 180.0,
            "movement_threshold_px": 20.0, "classes": [],
        },
    }


def _rules_to_dict(rules) -> dict:
    return {
        "detection": {
            "classes": list(rules.detection.classes),
            "min_confidence": rules.detection.min_confidence,
        },
        "zones": [
            {
                "name": z.name,
                "polygon": [list(p) for p in z.polygon],
                "rules": [
                    {"type": r.type, "classes": list(r.classes), "threshold": r.threshold}
                    for r in z.rules
                ],
            }
            for z in rules.zones
        ],
        "motion_gate": {
            "enabled": rules.motion_gate.enabled,
            "idle_seconds": rules.motion_gate.idle_seconds,
            "movement_threshold_px": rules.motion_gate.movement_threshold_px,
            "classes": list(rules.motion_gate.classes),
        },
    }


class _RulesUpdate(BaseModel):
    yaml: str


class _TrackingUpdate(BaseModel):
    enabled: bool


class _FpsUpdate(BaseModel):
    fps: int


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
            return _load_html()

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
                return {"path": None, "yaml": "", "parsed": _empty_parsed()}
            rules, _ = rs.current()
            return {
                "path": rs.path,
                "yaml": rs.read_yaml_text(),
                "parsed": _rules_to_dict(rules),
            }

        @app.get("/frame_size")
        def frame_size():
            cap = self._admin.get("capture_config")
            if cap is None:
                return {"width": 640, "height": 480}
            return {"width": cap.frame_width, "height": cap.frame_height}

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

        @app.get("/fps")
        def get_fps():
            cap = self._admin.get("capture_config")
            return {"fps": cap.fps if cap else 5}

        @app.post("/fps")
        def post_fps(body: _FpsUpdate):
            cap = self._admin.get("capture_config")
            if cap is None:
                raise HTTPException(503, "capture_config 미설정")
            if body.fps < 1 or body.fps > 60:
                raise HTTPException(400, "fps는 1~60 범위")
            cap.fps = body.fps
            return {"fps": cap.fps}

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
