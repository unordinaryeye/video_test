"""추론 워커. 입력 큐에서 프레임을 꺼내 모델 API로 보내고 결과를 출력 큐에 넣음.

룰(YAML)이 주어지면 detection 필터 + 구역 이벤트 + motion_gate를 적용한다.
"""

import logging
import queue
import threading
import time
from typing import Optional

from inference.client import ModelClient
from rules.detection_filter import filter_detections
from rules.zone_checker import check_zones
from rules.state import RulesState

logger = logging.getLogger("inference.worker")


class InferenceWorker:
    def __init__(
        self,
        input_queue: queue.Queue,
        output_queue: queue.Queue,
        client: ModelClient,
        rules_state: Optional[RulesState] = None,
    ):
        self._input = input_queue
        self._output = output_queue
        self._client = client
        self._rules_state = rules_state
        self._running = False
        self._thread = None
        self._processed = 0
        self._dropped = 0  # latest-only 정책으로 버려진 입력 프레임 수

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
        logger.info(
            f"Inference 워커 종료 (처리: {self._processed}프레임, 드롭: {self._dropped})"
        )

    def stats(self) -> dict:
        return {"processed": self._processed, "dropped": self._dropped}

    def _apply_rules(self, detections: list[dict]) -> tuple[list[dict], list[dict]]:
        """룰 적용. 반환: (필터링된 detections, zone_events dict 목록)."""
        if self._rules_state is None:
            return detections, []

        rules, motion_gate = self._rules_state.current()
        filtered = filter_detections(detections, rules.detection)

        idle_ids = motion_gate.update(filtered)
        filtered = motion_gate.filter(filtered, idle_ids)

        events = check_zones(filtered, rules.zones)
        events_dict = [
            {
                "zone_name": e.zone_name,
                "rule_type": e.rule_type,
                "detections": e.detections,
                "timestamp": e.timestamp.isoformat(),
            }
            for e in events
        ]
        return filtered, events_dict

    def _loop(self):
        while self._running:
            try:
                frame_data = self._input.get(timeout=1.0)
            except queue.Empty:
                continue

            # 실시간성 우선: 큐에 밀려있는 프레임이 있으면 최신 것만 처리.
            while True:
                try:
                    newer = self._input.get_nowait()
                except queue.Empty:
                    break
                # 직전에 잡고 있던 frame_data를 버리고 더 새 것으로 교체.
                self._dropped += 1
                frame_data = newer

            try:
                start = time.perf_counter()
                result = self._client.predict(frame_data)
                elapsed = (time.perf_counter() - start) * 1000

                detections = result.get("detections", [])
                filtered, zone_events = self._apply_rules(detections)
                result["detections"] = filtered
                result["zone_events"] = zone_events

                result["total_latency_ms"] = round(elapsed, 1)
                result["image_base64"] = frame_data["image_base64"]
                self._output.put(result)
                self._processed += 1

                if self._processed % 10 == 0:
                    logger.info(
                        f"추론 완료 #{self._processed} | "
                        f"frame_id={result.get('frame_id')} | "
                        f"latency={elapsed:.0f}ms | "
                        f"detections={len(filtered)} | events={len(zone_events)}"
                    )
            except Exception as e:
                logger.error(f"추론 실패 frame_id={frame_data.get('frame_id')}: {e}")
