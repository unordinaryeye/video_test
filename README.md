# 실시간 영상 추론 파이프라인 프로토타입

실시간 영상 처리 파이프라인 프로토타입입니다.
**영상 수신 → 모델 추론 → 결과 전송**의 3단계 파이프라인을 독립적으로 분리하여 구성했으며,
각 컴포넌트(영상 소스/모델/결과 전송)를 설정만 바꾸어 교체할 수 있도록 설계되었습니다.

## 목적

- 회사 내부 플랫폼에서 "실시간 영상 추론이 가능하다"는 것을 시각적으로 보여주기 위한 데모
- 각 파이프라인 단계가 **눈에 보이는** 프로토타입 구조
- 실제 환경으로 전환할 때 **소스/모델/Sink 교체가 쉬운 구조**

---

## 아키텍처

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ 영상 소스    │    │ 추론 API 서버    │    │ 결과 저장소     │
│ (Webcam,     │    │ (FastAPI+YOLOv8) │    │ (Console/File/  │
│  File, RTSP, │    │   GPU/CPU        │    │  Web/Kafka)     │
│  Synthetic)  │    │                  │    │                 │
└──────┬───────┘    └────────▲─────────┘    └────────▲────────┘
       │                     │                       │
       │ OpenCV              │ HTTP (JSON+base64)    │
       ▼                     │                       │
┌─────────────┐  Queue  ┌────┴──────┐  Queue  ┌──────┴──────┐
│  Capture    │────────▶│ Inference │────────▶│    Sink     │
│  (수신)     │  frame  │  (추론)   │  result │   (전송)    │
└─────────────┘         └───────────┘         └─────────────┘
      capture/             inference/               sink/
```

### 통신 방식

- **Capture → Inference**: `queue.Queue` (스레드 간, 실시간 특성상 큐가 가득 차면 오래된 프레임 드롭)
- **Inference → Model API**: HTTP POST (이미지는 base64 JPEG로 인코딩)
- **Inference → Sink**: `queue.Queue`
- **Sink → 외부**: Console 출력 / File(JSONL) / Web(MJPEG 스트림) / Kafka Producer

---

## 디렉토리 구조

```
video_test/
├── config.py                # 모든 설정 한 곳에 집중 (교체 포인트)
├── pipeline.py              # 메인 오케스트레이터
├── docker-compose.yml       # model-server + pipeline 2컨테이너
├── Dockerfile               # pipeline 컨테이너 이미지
├── requirements.txt
├── README.md
│
├── capture/
│   └── capture.py           # 4종 영상 소스: webcam / file / rtsp / synthetic
│
├── inference/
│   ├── client.py            # 모델 API 호출 클라이언트 (재시도 포함)
│   └── worker.py            # 큐 기반 추론 워커
│
├── sink/
│   ├── base.py              # Sink 인터페이스 (Protocol)
│   ├── console_sink.py      # 콘솔 출력 (개발/데모용)
│   ├── file_sink.py         # JSONL 파일 저장
│   ├── web_sink.py          # FastAPI 기반 웹 뷰어 (MJPEG 스트림)
│   ├── kafka_sink.py        # Kafka Producer (운영용)
│   ├── multi_sink.py        # 여러 Sink 동시 사용 래퍼
│   └── worker.py            # 큐 기반 Sink 워커
│
├── model_server/
│   ├── Dockerfile           # CUDA 베이스 이미지
│   └── app.py               # FastAPI 추론 서버 (YOLOv8n)
│
├── tests/
└── results/                 # FileSink 출력 디렉토리 (자동 생성)
```

---

## 사전 요구사항

### 공통
- Docker Desktop (WSL2 백엔드)
- NVIDIA 드라이버 (GPU 사용 시)

### 로컬 실행 시 추가
- Python 3.11 이상
- 웹캠 (로컬 실행 + 웹캠 소스 사용 시)

### 검증된 환경
- Windows 11 Pro
- Docker Desktop (WSL2)
- NVIDIA GTX 1660 Ti (VRAM 6GB)
- Python 3.11

---

## 실행 방법

### 방법 1: Docker Compose로 전체 실행 (합성 프레임)

가장 간단한 방법. 가짜 프레임(움직이는 사각형)으로 파이프라인 전체 흐름을 검증합니다.

```bash
docker compose up --build
```

- 모델 서버: http://localhost:8001
- 웹 뷰어: http://localhost:8080

종료:
```bash
docker compose down
```

### 방법 2: Docker(모델) + 로컬(파이프라인) — **웹캠 데모 권장**

웹캠은 Windows Docker에서 접근이 까다로워 pipeline만 로컬에서 실행합니다.

```bash
# 1. 모델 서버는 Docker에 유지
docker compose up -d model-server

# 2. 로컬 Python 의존성 설치 (최초 1회만)
pip install opencv-python httpx numpy Pillow fastapi uvicorn

# 3. 파이프라인을 로컬에서 실행 (웹캠 소스 + 웹 뷰어)
python pipeline.py --source webcam --source-uri 0 \
                   --model-url http://localhost:8001/predict \
                   --sink console,web
```

브라우저에서 **http://localhost:8080** 열면 실시간 영상 + 감지 결과가 보입니다.
종료는 `Ctrl+C`. 자세한 백그라운드 실행/종료 방법은 아래 [실행 및 종료 명령어](#실행-및-종료-명령어) 참조.

### 방법 3: 샘플 MP4 파일로 실행

```bash
python pipeline.py --source file --source-uri samples/sample.mp4 \
                   --model-url http://localhost:8001/predict \
                   --sink console,web
```

파일이 끝나면 자동으로 처음부터 반복 재생됩니다.

---

## 실행 및 종료 명령어

웹캠 데모(방법 2)를 기준으로 포그라운드/백그라운드 실행과 종료 방법을 정리합니다.
모든 명령은 프로젝트 폴더에서 실행합니다.

```bash
cd /path/to/video_test
```

### 1단계: 모델 서버 실행 (Docker, 백그라운드)

```bash
docker compose up -d model-server
```

- `-d` 옵션으로 백그라운드 detached 모드 실행
- 최초 실행 시 YOLOv8 모델 다운로드로 10초 정도 소요
- 준비 상태 확인:
  ```bash
  curl http://localhost:8001/health
  # {"status":"ok","model_loaded":true}
  ```

### 2단계: 파이프라인 실행 (로컬 Python)

#### 옵션 A — 포그라운드 (로그 실시간 확인)

```bash
python pipeline.py --source webcam --source-uri 0 \
                   --model-url http://localhost:8001/predict \
                   --sink console,web
```

종료: `Ctrl+C`

#### 옵션 B — 백그라운드 + 로그 파일 (Git Bash / WSL)

```bash
nohup python pipeline.py --source webcam --source-uri 0 \
  --model-url http://localhost:8001/predict \
  --sink console,web > pipeline.log 2>&1 &
```

- `nohup ... &` : 터미널을 닫아도 계속 실행
- `> pipeline.log 2>&1` : 표준출력/에러를 `pipeline.log`로 저장
- 실행 중 로그 확인:
  ```bash
  tail -f pipeline.log
  ```

#### 옵션 C — 백그라운드 (PowerShell)

```powershell
Start-Process python `
  -ArgumentList "pipeline.py","--source","webcam","--source-uri","0",`
                "--model-url","http://localhost:8001/predict",`
                "--sink","console,web" `
  -RedirectStandardOutput pipeline.log `
  -RedirectStandardError pipeline.err `
  -NoNewWindow
```

### 3단계: 웹 뷰어 확인

브라우저에서 **http://localhost:8080** 접속.

### 4단계: 종료 방법

#### 파이프라인 종료

**포그라운드로 실행 중이면**: 해당 터미널에서 `Ctrl+C`

**Git Bash / WSL 백그라운드 종료**:
```bash
# 현재 셸의 백그라운드 작업 목록 확인
jobs
# 첫 번째 작업 종료
kill %1
```

**PowerShell 백그라운드 종료 (포트 기준)**:
```powershell
# 8080 포트를 점유한 PID 찾기
netstat -ano | findstr :8080
# 해당 PID 종료 (예: 12345)
taskkill /PID 12345 /F
```

**PowerShell 백그라운드 종료 (프로세스 기준)**:
```powershell
Get-Process python | Stop-Process
```

#### 모델 서버 종료

```bash
docker compose down
```

### 전체 재시작 한 줄 (Git Bash)

```bash
docker compose down && docker compose up -d model-server && sleep 5 && \
python pipeline.py --source webcam --source-uri 0 \
  --model-url http://localhost:8001/predict --sink console,web
```

### 상태 확인 명령

```bash
# Docker 컨테이너 상태
docker compose ps

# 8080(뷰어)·8001(모델)·8080 점유 프로세스
netstat -ano | findstr "8001 8080"

# 현재 통계 API 호출
curl http://localhost:8080/stats
```

---

## 웹 뷰어 사용법

http://localhost:8080 에서 확인할 수 있는 정보:

| 영역 | 내용 |
|------|------|
| 좌측 영상 | 실시간 MJPEG 스트림 + 감지 박스 오버레이 (프레임 번호/지연시간 포함) |
| 파이프라인 상태 | Capture → Inference → Sink 3단계 시각화 |
| 통계 카드 | 처리 프레임, 최근 감지 수, 추론 지연(ms), FPS |
| 감지 로그 | 최근 20개의 감지 이벤트 (프레임 번호 + 라벨) |

### API 엔드포인트

| 경로 | 역할 |
|------|------|
| `GET /` | 메인 HTML 페이지 |
| `GET /stream` | MJPEG 비디오 스트림 |
| `GET /stats` | JSON 통계 (500ms 폴링용) |

---

## 교체 포인트 (실제 운영 환경으로 전환 시)

### 영상 소스 교체

`capture/capture.py`의 `VideoSource` 구현체 중 하나를 선택하거나 새로 추가합니다.

| source_type | 설명 | source_uri 예시 |
|-------------|------|-----------------|
| `webcam` | 로컬 웹캠 | `"0"` (디바이스 ID) |
| `file` | 영상 파일 반복 재생 | `"samples/sample.mp4"` |
| `rtsp` | RTSP 스트림 (공장 CCTV 등) | `"rtsp://192.168.1.100:554/stream"` |
| `synthetic` | 합성 프레임 (테스트용) | - |

새 소스 추가 시 `VideoSource`를 상속하여 `read()`만 구현하면 됩니다.

### 모델 교체

`model_server/app.py`의 `load_model()` 함수만 수정하면 됩니다.

```python
def load_model():
    global _model
    from ultralytics import YOLO
    _model = YOLO("yolov8n.pt")  # ← 이 줄만 변경
    # 예: _model = YOLO("custom_safety_model.pt")
    return _model
```

**요청/응답 스키마**는 그대로 유지하거나 `PredictResponse`를 조정할 수 있습니다.

### 결과 전송(Sink) 교체

`docker-compose.yml` 또는 CLI에서 `SINK_TYPE`만 변경합니다.

| sink_type | 용도 |
|-----------|------|
| `console` | 개발/디버깅 |
| `file` | 결과 JSONL 저장 (`results/output.jsonl`) |
| `web` | 실시간 웹 뷰어 (MJPEG) |
| `kafka` | 운영 환경 메시지 브로커 전송 |

**여러 Sink 동시 사용**은 콤마로 구분:
```bash
SINK_TYPE=console,file,web
```

새 Sink는 `sink/` 아래에 추가하고 `sink/base.py`의 프로토콜(`send`, `close`)을 구현합니다.

---

## 설정 (config.py)

모든 기본값은 `config.py`의 dataclass에 모여 있습니다. 우선순위:

```
CLI 인자 > 환경 변수 > config.py 기본값
```

| 환경변수 | CLI 인자 | 설명 |
|----------|---------|------|
| `CAPTURE_SOURCE_TYPE` | `--source` | 영상 소스 종류 |
| `CAPTURE_SOURCE_URI` | `--source-uri` | 소스 경로/ID |
| `CAPTURE_FPS` | `--fps` | 프레임 추출 속도 |
| `MODEL_API_URL` | `--model-url` | 추론 API 엔드포인트 |
| `SINK_TYPE` | `--sink` | 결과 전송 종류 (콤마 구분) |

---

## GPU 사용 관련

현재 구성은 GPU를 자동 사용하도록 되어 있으나, **PyTorch가 요구하는 CUDA 버전과 호스트 드라이버가 맞지 않으면 CPU로 폴백**됩니다.

- 호스트 NVIDIA 드라이버가 요구 버전보다 낮으면 다음 경고가 나옵니다:
  ```
  CUDA initialization: The NVIDIA driver on your system is too old
  ```
- 이 경우 `model_server/Dockerfile`에서 PyTorch를 호환 버전으로 고정해야 합니다.
- CPU 폴백 시에도 YOLOv8n 기준 200~1000ms에 추론이 완료되어 **데모 용도로는 충분**합니다.

### GPU 활성화 (선택)

`model_server/Dockerfile`에서 다음과 같이 호환 PyTorch를 설치하도록 수정:

```dockerfile
RUN pip3 install --no-cache-dir \
    torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir \
    fastapi uvicorn ultralytics>=8.3.0 ...
```

---

## 트러블슈팅

### 포트 충돌 (`Bind for 0.0.0.0:8001 failed`)

다른 서비스가 8001을 쓰고 있을 경우 `docker-compose.yml`의 포트 매핑을 변경:
```yaml
model-server:
  ports:
    - "8002:8000"  # ← 변경
```
`MODEL_API_URL`도 함께 갱신.

### Dockerfile 빌드 실패 (`libgl1-mesa-glx`)

Debian Trixie에서 제거됨. `libgl1` 로 사용.

### PyTorch 로드 실패 (`Weights only load failed`)

ultralytics 8.3.0 이상으로 업그레이드. 현재 `requirements.txt`는 `ultralytics>=8.3.0`으로 고정.

### 웹캠 초기화 지연

Windows에서 OpenCV가 웹캠을 여는 데 20~30초 걸릴 수 있습니다. 첫 프레임이 늦게 나오는 건 정상입니다.

### Docker 컨테이너에서 웹캠 접근

Windows Docker Desktop(WSL2)은 USB 카메라를 기본 지원하지 않습니다.
**웹캠 데모는 방법 2(로컬+Docker 하이브리드)** 로 진행하세요.

### Docker 로그에 콘솔 출력이 안 보일 때

`PYTHONUNBUFFERED=1` 환경변수를 Dockerfile에 추가 (현재 설정 완료됨).

---

## 파이프라인 컴포넌트 요약

| 컴포넌트 | 파일 | 역할 |
|---------|------|------|
| FrameCapture | `capture/capture.py` | 영상 소스에서 프레임을 주기적으로 읽어 큐에 투입 |
| ModelClient | `inference/client.py` | 추론 API 호출 (재시도/타임아웃 포함) |
| InferenceWorker | `inference/worker.py` | 큐에서 프레임을 꺼내 API 호출, 결과를 다음 큐에 전달 |
| SinkWorker | `sink/worker.py` | 큐에서 결과를 꺼내 선택된 Sink로 전송 |
| WebSink | `sink/web_sink.py` | FastAPI로 HTML 페이지 + MJPEG 스트림 제공 |
| ModelServer | `model_server/app.py` | FastAPI + YOLOv8n. `/predict`, `/health` 엔드포인트 |

각 워커는 독립적인 **데몬 스레드**로 실행되며, 큐를 통해 느슨하게 결합되어 있어
한 단계가 느려져도 전체가 멈추지 않습니다. 큐가 가득 차면 오래된 프레임은 드롭됩니다.

---

## 라이선스 및 참고

- YOLOv8: [Ultralytics](https://docs.ultralytics.com/) (AGPL-3.0)
- 본 프로토타입은 내부 데모 및 파이프라인 구조 검증 목적으로 제작됨
