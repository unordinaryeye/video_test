"""sink/worker.py — latest-only 정책의 드롭 카운터 단위 테스트."""

import queue
import threading
import time

from sink.worker import SinkWorker


class _SlowSink:
    """send()가 일정 시간을 소비하고 받은 항목을 기록하는 모의 sink."""

    def __init__(self, delay_sec: float = 0.05):
        self.delay = delay_sec
        self.received: list[dict] = []
        self._lock = threading.Lock()

    def send(self, result: dict) -> None:
        time.sleep(self.delay)
        with self._lock:
            self.received.append(result)

    def close(self) -> None:
        pass


def test_stats_initial_values():
    """초기 상태에서 sent=0, dropped=0."""
    q_in = queue.Queue(maxsize=100)
    worker = SinkWorker(q_in, _SlowSink(delay_sec=0.0))
    assert worker.stats() == {"sent": 0, "dropped": 0}


def test_latest_only_drops_intermediate_results():
    """결과 큐에 여러 항목이 쌓이면 워커는 최신만 sink에 전달."""
    q_in = queue.Queue(maxsize=100)
    sink = _SlowSink(delay_sec=0.1)
    worker = SinkWorker(q_in, sink)

    for i in range(10):
        q_in.put({"frame_id": i})

    worker.start()
    time.sleep(0.5)
    worker.stop()

    stats = worker.stats()
    assert stats["sent"] >= 1
    assert stats["dropped"] >= 1, f"드롭이 0이면 latest-only 로직이 동작하지 않음: {stats}"
    # 마지막으로 sink에 도달한 항목은 가장 큰 frame_id여야 함.
    assert sink.received[-1]["frame_id"] == 9


def test_no_drop_when_results_arrive_slowly():
    """천천히 들어오는 결과는 드롭 없이 모두 전달."""
    q_in = queue.Queue(maxsize=100)
    sink = _SlowSink(delay_sec=0.0)
    worker = SinkWorker(q_in, sink)
    worker.start()

    for i in range(3):
        q_in.put({"frame_id": i})
        time.sleep(0.05)

    time.sleep(0.1)
    worker.stop()

    stats = worker.stats()
    assert stats["dropped"] == 0, f"여유 있는 입력에서 드롭이 발생: {stats}"
    assert stats["sent"] == 3
    assert [r["frame_id"] for r in sink.received] == [0, 1, 2]
