"""파일 저장 Sink. 테스트/분석용. JSONL 형식."""

import json
import logging
import os

logger = logging.getLogger("sink.file")


class FileSink:
    def __init__(self, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self._file = open(output_path, "a", encoding="utf-8")
        logger.info(f"FileSink 열림: {output_path}")

    def send(self, result: dict) -> None:
        self._file.write(json.dumps(result, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()
        logger.info("FileSink 닫힘")
