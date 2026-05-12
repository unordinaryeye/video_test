"""Sink 워커. 결과 큐에서 데이터를 꺼내 선택된 Sink로 전송."""

import logging
import queue
import threading

logger = logging.getLogger("sink.worker")


class SinkWorker:
    def __init__(self, input_queue: queue.Queue, sink):
        self._input = input_queue
        self._sink = sink
        self._running = False
        self._thread = None
        self._sent = 0
        self._dropped = 0  # latest-only 정책으로 버려진 결과 수

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Sink 워커 시작")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self._sink.close()
        logger.info(f"Sink 워커 종료 (전송: {self._sent}건, 드롭: {self._dropped})")

    def stats(self) -> dict:
        return {"sent": self._sent, "dropped": self._dropped}

    def _loop(self):
        while self._running:
            try:
                result = self._input.get(timeout=1.0)
            except queue.Empty:
                continue

            # 실시간성 우선: 큐에 밀려있는 결과가 있으면 최신 것만 취함.
            # 뒤처진 중간 프레임은 버려서 지연 누적 방지.
            while True:
                try:
                    newer = self._input.get_nowait()
                except queue.Empty:
                    break
                self._dropped += 1
                result = newer

            try:
                self._sink.send(result)
                self._sent += 1
            except Exception as e:
                logger.error(f"Sink 전송 실패: {e}")
