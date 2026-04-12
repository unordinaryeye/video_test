"""콘솔 출력 Sink. 개발/데모용."""

import json
import logging

logger = logging.getLogger("sink.console")


class ConsoleSink:
    def send(self, result: dict) -> None:
        detections = result.get("detections", [])
        frame_id = result.get("frame_id", "?")
        latency = result.get("inference_time_ms", 0)
        count = len(detections)

        labels = ", ".join(d["label"] for d in detections[:5])
        if count > 5:
            labels += f" (+{count - 5})"

        print(
            f"[Frame#{frame_id}] "
            f"감지: {count}개 ({labels}) | "
            f"추론: {latency:.0f}ms",
            flush=True,
        )

    def close(self) -> None:
        pass
