"""inference/worker.py — latest-only 정책의 드롭 카운터 단위 테스트."""

import queue
import time

import pytest

from inference.worker import InferenceWorker


class _SlowClient:
    """predict()가 일정 시간을 소비하는 모의 클라이언트."""

    def __init__(self, delay_sec: float = 0.05):
        self.delay = delay_sec
        self.calls = 0

    def predict(self, frame_data: dict) -> dict:
        self.calls += 1
        time.sleep(self.delay)
        return {
            "frame_id": frame_data["frame_id"],
            "detections": [],
            "inference_time_ms": self.delay * 1000,
        }

    def close(self):
        pass


def _frame(i: int) -> dict:
    return {"frame_id": i, "image_base64": "x", "timestamp": ""}


def test_processed_and_dropped_counts_in_stats():
    """워커는 stats()로 처리/드롭 수를 노출해야 한다."""
    in_q, out_q = queue.Queue(maxsize=100), queue.Queue(maxsize=100)
    worker = InferenceWorker(in_q, out_q, _SlowClient(delay_sec=0.0))
    assert worker.stats() == {"processed": 0, "dropped": 0}


def test_latest_only_drops_intermediate_frames():
    """입력 큐에 여러 프레임이 쌓이면 워커는 가장 최신만 처리하고 나머지는 드롭."""
    in_q, out_q = queue.Queue(maxsize=100), queue.Queue(maxsize=100)
    client = _SlowClient(delay_sec=0.1)  # 추론이 캡처보다 느린 상황 모사
    worker = InferenceWorker(in_q, out_q, client)

    # 워커가 첫 항목 처리하는 동안 9개를 더 밀어넣음.
    for i in range(10):
        in_q.put(_frame(i))

    worker.start()
    # 한 사이클이 끝날 시간 (0.1s 추론 + 여유) 충분히 대기.
    time.sleep(0.5)
    worker.stop()

    stats = worker.stats()
    # 첫 get() 후 latest-only 루프가 나머지를 빨아들였어야 함.
    assert stats["processed"] >= 1
    assert stats["dropped"] >= 1, f"드롭이 0이면 latest-only 로직이 동작하지 않음: {stats}"
    # 처리한 마지막 프레임은 가장 큰 frame_id (드롭 후 최신만 남음).
    assert not out_q.empty()
    result = out_q.get_nowait()
    assert result["frame_id"] == 9, "최신 프레임이 아닌 것이 처리됨"


def test_no_drop_when_queue_one_at_a_time():
    """프레임이 천천히 들어오면 드롭은 0이어야 한다."""
    in_q, out_q = queue.Queue(maxsize=100), queue.Queue(maxsize=100)
    worker = InferenceWorker(in_q, out_q, _SlowClient(delay_sec=0.0))
    worker.start()

    for i in range(3):
        in_q.put(_frame(i))
        time.sleep(0.05)  # 워커가 비울 시간 확보

    time.sleep(0.1)
    worker.stop()

    stats = worker.stats()
    assert stats["dropped"] == 0, f"여유 있는 입력에서 드롭이 발생: {stats}"
    assert stats["processed"] == 3
