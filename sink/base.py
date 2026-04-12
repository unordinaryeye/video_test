"""Sink 인터페이스. 새 Sink를 추가하려면 이 프로토콜을 구현."""

from typing import Protocol


class Sink(Protocol):
    def send(self, result: dict) -> None:
        """추론 결과를 전송."""
        ...

    def close(self) -> None:
        """리소스 정리."""
        ...
