"""추론 워커. 입력 큐에서 프레임을 꺼내 모델 API로 보내고 결과를 출력 큐에 넣음."""

import logging
import queue
import threading
import time

from inference.client import ModelClient

logger = logging.getLogger("inference.worker")


class InferenceWorker:
    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        client: ModelClient,
    ):
        self._input = input_queue
        self._output = output_queue
        self._client = client
        self._running = False
        self._thread = None
        self._processed = 0

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Inference 워커 시작")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self._client.close()
        logger.info(f"Inference 워커 종료 (처리: {self._processed}프레임)")

    def _loop(self):
        while self._running:
            try:
                frame_data = self._input.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                start = time.perf_counter()
                result = self._client.predict(frame_data)
                elapsed = (time.perf_counter() - start) * 1000

                result["total_latency_ms"] = round(elapsed, 1)
                # WebSink가 박스를 그리기 위해 원본 프레임도 함께 전달
                result["image_base64"] = frame_data["image_base64"]
                self._output.put(result)
                self._processed += 1

                if self._processed % 10 == 0:
                    logger.info(
                        f"추론 완료 #{self._processed} | "
                        f"frame_id={result.get('frame_id')} | "
                        f"latency={elapsed:.0f}ms"
                    )
            except Exception as e:
                logger.error(f"추론 실패 frame_id={frame_data.get('frame_id')}: {e}")
