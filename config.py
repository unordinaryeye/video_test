"""파이프라인 전체 설정. 영상 소스와 모델을 여기서 교체."""

from dataclasses import dataclass, field


@dataclass
class CaptureConfig:
    """영상 소스 설정. source_type만 바꾸면 소스 교체."""
    source_type: str = "webcam"  # "webcam" | "file" | "rtsp" | "synthetic"
    source_uri: str = "0"  # 웹캠: "0", 파일: "path.mp4", RTSP: "rtsp://..."
    fps: int = 5
    frame_width: int = 640
    frame_height: int = 480


@dataclass
class ModelConfig:
    """모델 API 설정. api_url만 바꾸면 모델 교체."""
    api_url: str = "http://model-server:8000/predict"
    health_url: str = "http://model-server:8000/health"
    timeout_sec: float = 5.0
    max_retries: int = 3


@dataclass
class SinkConfig:
    """결과 전송 설정. sink_type만 바꾸면 전송 대상 교체."""
    sink_type: str = "console"  # "console" | "file" | "kafka"
    # File sink
    output_path: str = "results/output.jsonl"
    # Kafka sink
    kafka_bootstrap_servers: str = "kafka:9092"
    kafka_topic: str = "inference-results"


@dataclass
class PipelineConfig:
    """파이프라인 전체 설정."""
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sink: SinkConfig = field(default_factory=SinkConfig)
    queue_max_size: int = 30
    log_level: str = "INFO"
