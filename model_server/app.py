"""FastAPI 추론 서버. 모델을 교체하려면 load_model()만 수정."""

import base64
import io
import logging
import threading
import time

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="Safety Inference Server")
logger = logging.getLogger("model-server")

# --- 모델 로딩 (여기만 바꾸면 모델 교체) ---
_model = None
_predict_lock = threading.Lock()


def load_model():
    """모델 로드. 다른 모델로 바꾸려면 이 함수만 수정."""
    global _model
    from ultralytics import YOLO
    _model = YOLO("yolov8n.pt")  # nano 모델, ~6MB, VRAM ~1GB
    logger.info("YOLOv8n 모델 로드 완료")
    return _model


def get_model():
    global _model
    if _model is None:
        load_model()
    return _model


# --- 요청/응답 스키마 ---
class PredictRequest(BaseModel):
    image_base64: str
    frame_id: int = 0
    timestamp: str = ""
    use_tracking: bool = False


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: list[float]  # [x1, y1, x2, y2]
    track_id: int | None = None


class PredictResponse(BaseModel):
    frame_id: int
    detections: list[Detection]
    inference_time_ms: float


# --- 엔드포인트 ---
@app.on_event("startup")
async def startup():
    logger.info("모델 서버 시작, 모델 로딩 중...")
    load_model()
    logger.info("모델 서버 준비 완료")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/classes")
async def classes():
    model = get_model()
    return {"classes": list(model.names.values())}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        image_bytes = base64.b64decode(req.image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        frame = np.array(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 디코딩 실패: {e}")

    model = get_model()

    with _predict_lock:
        start = time.perf_counter()
        if req.use_tracking:
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
        else:
            results = model(frame, verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000

    detections = []
    for r in results:
        for box in r.boxes:
            track_id = None
            if req.use_tracking and box.id is not None:
                track_id = int(box.id[0])
            detections.append(Detection(
                label=r.names[int(box.cls[0])],
                confidence=round(float(box.conf[0]), 3),
                bbox=[round(float(c), 1) for c in box.xyxy[0]],
                track_id=track_id,
            ))

    return PredictResponse(
        frame_id=req.frame_id,
        detections=detections,
        inference_time_ms=round(elapsed_ms, 1),
    )
