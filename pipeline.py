"""파이프라인 오케스트레이터. 3단계를 큐로 연결하고 실행.

사용법:
  # 기본 (합성 프레임 + 콘솔 출력)
  python pipeline.py

  # 웹캠 + 파일 저장
  python pipeline.py --source webcam --sink file

  # 환경변수로도 설정 가능 (Docker용)
  CAPTURE_SOURCE_TYPE=synthetic MODEL_API_URL=http://localhost:8000/predict python pipeline.py
"""

import argparse
import logging
import os
import queue
import signal
import sys
import time

from config import PipelineConfig, CaptureConfig, ModelConfig, SinkConfig
from capture.capture import FrameCapture
from inference.client import ModelClient
from inference.worker import InferenceWorker
from rules.loader import load_rules
from sink.worker import SinkWorker
from sink.console_sink import ConsoleSink
from sink.file_sink import FileSink
from sink.multi_sink import MultiSink


def _create_one_sink(sink_type: str, config: SinkConfig):
    """단일 Sink 생성."""
    sink_type = sink_type.strip()
    if sink_type == "console":
        return ConsoleSink()
    elif sink_type == "file":
        return FileSink(config.output_path)
    elif sink_type == "web":
        from sink.web_sink import WebSink
        return WebSink(port=8080)
    elif sink_type == "kafka":
        from sink.kafka_sink import KafkaSink
        return KafkaSink(config.kafka_bootstrap_servers, config.kafka_topic)
    else:
        raise ValueError(f"지원하지 않는 sink_type: {sink_type}")


def create_sink(config: SinkConfig):
    """sink_type에 따라 Sink 생성. 콤마로 구분하면 여러 Sink 동시 사용."""
    types = [t for t in config.sink_type.split(",") if t.strip()]
    if len(types) == 1:
        return _create_one_sink(types[0], config)
    return MultiSink([_create_one_sink(t, config) for t in types])


def build_config(args) -> PipelineConfig:
    """CLI 인자 + 환경변수로 설정 구성."""
    config = PipelineConfig()

    # CLI 인자 우선, 없으면 환경변수, 없으면 기본값
    config.capture.source_type = (
        args.source
        or os.environ.get("CAPTURE_SOURCE_TYPE")
        or config.capture.source_type
    )
    config.capture.source_uri = (
        args.source_uri
        or os.environ.get("CAPTURE_SOURCE_URI")
        or config.capture.source_uri
    )
    config.capture.fps = int(
        args.fps
        or os.environ.get("CAPTURE_FPS")
        or config.capture.fps
    )
    config.model.api_url = (
        args.model_url
        or os.environ.get("MODEL_API_URL")
        or config.model.api_url
    )
    config.sink.sink_type = (
        args.sink
        or os.environ.get("SINK_TYPE")
        or config.sink.sink_type
    )
    config.rules.rules_path = (
        args.rules
        or os.environ.get("RULES_PATH")
        or config.rules.rules_path
    )
    if args.tracking or os.environ.get("USE_TRACKING", "").lower() in ("1", "true", "yes"):
        config.model.use_tracking = True

    return config


def main():
    parser = argparse.ArgumentParser(description="실시간 영상 추론 파이프라인")
    parser.add_argument("--source", choices=["webcam", "file", "rtsp", "synthetic"])
    parser.add_argument("--source-uri", dest="source_uri")
    parser.add_argument("--fps", type=int)
    parser.add_argument("--model-url", dest="model_url")
    parser.add_argument("--sink", help="console | file | web | kafka (콤마로 구분 가능)")
    parser.add_argument("--rules", help="YAML 룰 파일 경로 (예: config/rules.yaml)")
    parser.add_argument("--tracking", action="store_true", help="객체 트래킹 활성화 (track_id 부여)")
    args = parser.parse_args()

    config = build_config(args)

    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("pipeline")

    # --- 파이프라인 구성 ---
    frame_queue = queue.Queue(maxsize=config.queue_max_size)
    result_queue = queue.Queue(maxsize=config.queue_max_size)

    logger.info("=" * 60)
    logger.info("파이프라인 시작")
    logger.info(f"  영상 소스: {config.capture.source_type} ({config.capture.source_uri})")
    logger.info(f"  모델 API: {config.model.api_url}")
    logger.info(f"  결과 전송: {config.sink.sink_type}")
    logger.info(f"  트래킹: {config.model.use_tracking}")
    logger.info(f"  룰 파일: {config.rules.rules_path or '(없음, 필터링 비활성)'}")
    logger.info("=" * 60)

    # 0. 룰 로드
    rules = load_rules(config.rules.rules_path)
    if config.rules.rules_path:
        logger.info(
            f"룰 로드 완료 — classes={rules.detection.classes}, "
            f"min_conf={rules.detection.min_confidence}, "
            f"zones={len(rules.zones)}, motion_gate={rules.motion_gate.enabled}"
        )

    # 1. 모델 클라이언트
    client = ModelClient(config.model)

    # 모델 서버 대기
    logger.info("모델 서버 연결 대기 중...")
    for i in range(30):
        if client.health_check():
            logger.info("모델 서버 연결 성공")
            break
        time.sleep(2)
    else:
        logger.error("모델 서버 연결 실패. 서버가 실행 중인지 확인하세요.")
        sys.exit(1)

    # 2. 컴포넌트 생성
    capture = FrameCapture(config.capture, frame_queue)
    inference = InferenceWorker(frame_queue, result_queue, client, rules=rules)
    sink = SinkWorker(result_queue, create_sink(config.sink))

    # 3. 시작 (순서: sink -> inference -> capture)
    sink.start()
    inference.start()
    capture.start()

    # 4. Ctrl+C로 종료
    def shutdown(sig, frame):
        logger.info("종료 신호 수신, 파이프라인 정리 중...")
        capture.stop()
        inference.stop()
        sink.stop()
        logger.info("파이프라인 종료 완료")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # 메인 스레드 유지
    logger.info("파이프라인 실행 중... (Ctrl+C로 종료)")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
