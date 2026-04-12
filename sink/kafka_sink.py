"""Kafka 전송 Sink. 운영 환경용."""

import json
import logging

logger = logging.getLogger("sink.kafka")


class KafkaSink:
    def __init__(self, bootstrap_servers: str, topic: str):
        try:
            from confluent_kafka import Producer
        except ImportError:
            raise ImportError(
                "confluent-kafka 패키지가 필요합니다. "
                "pip install confluent-kafka 로 설치하세요."
            )

        self._producer = Producer({"bootstrap.servers": bootstrap_servers})
        self._topic = topic
        logger.info(f"KafkaSink 연결: {bootstrap_servers}, topic={topic}")

    def send(self, result: dict) -> None:
        payload = json.dumps(result, ensure_ascii=False).encode("utf-8")
        self._producer.produce(
            self._topic,
            value=payload,
            callback=self._delivery_callback,
        )
        self._producer.poll(0)

    def close(self) -> None:
        self._producer.flush(timeout=5)
        logger.info("KafkaSink 닫힘")

    @staticmethod
    def _delivery_callback(err, msg):
        if err:
            logger.error(f"Kafka 전송 실패: {err}")
