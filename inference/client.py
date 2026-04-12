"""모델 API 호출 클라이언트. api_url만 바꾸면 다른 모델 서버로 전환."""

import logging
import time

import httpx

from config import ModelConfig

logger = logging.getLogger("inference.client")


class ModelClient:
    """모델 API 서버에 추론 요청을 보내는 클라이언트."""

    def __init__(self, config: ModelConfig):
        self._config = config
        self._client = httpx.Client(timeout=config.timeout_sec)
        # api_url에서 health_url 자동 유도 (둘이 어긋나지 않게)
        self._health_url = config.api_url.rsplit("/", 1)[0] + "/health"

    def health_check(self) -> bool:
        try:
            resp = self._client.get(self._health_url)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def predict(self, frame_data: dict) -> dict:
        """프레임 데이터를 API로 보내고 추론 결과를 받음."""
        payload = {
            "image_base64": frame_data["image_base64"],
            "frame_id": frame_data["frame_id"],
            "timestamp": frame_data["timestamp"],
        }

        last_error = None
        for attempt in range(1, self._config.max_retries + 1):
            try:
                resp = self._client.post(
                    self._config.api_url,
                    json=payload,
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPError as e:
                last_error = e
                logger.warning(f"추론 요청 실패 (시도 {attempt}/{self._config.max_retries}): {e}")
                time.sleep(0.5 * attempt)

        logger.error(f"추론 요청 최종 실패: {last_error}")
        raise last_error

    def close(self):
        self._client.close()
