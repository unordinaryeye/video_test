"""여러 Sink로 동시에 전송하는 래퍼."""

import logging

logger = logging.getLogger("sink.multi")


class MultiSink:
    """여러 Sink에 동일한 결과를 브로드캐스트."""

    def __init__(self, sinks: list):
        self._sinks = sinks
        logger.info(f"MultiSink: {len(sinks)}개 Sink 연결됨")

    def send(self, result: dict) -> None:
        for sink in self._sinks:
            try:
                sink.send(result)
            except Exception as e:
                logger.error(f"{type(sink).__name__} 전송 실패: {e}")

    def close(self) -> None:
        for sink in self._sinks:
            try:
                sink.close()
            except Exception as e:
                logger.error(f"{type(sink).__name__} 종료 실패: {e}")
