FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# PyTorch를 CUDA 12.4 호환 버전으로 먼저 고정 설치 (호스트 드라이버 550+ 요구).
# 이 단계를 건너뛰면 ultralytics가 최신 cu130 torch를 끌어와서 드라이버 불일치 발생.
RUN pip3 install --no-cache-dir \
    torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip3 install --no-cache-dir \
    fastapi==0.111.0 \
    uvicorn==0.30.1 \
    ultralytics>=8.3.0 \
    python-multipart==0.0.9 \
    numpy>=1.26.0 \
    Pillow>=10.0.0

COPY model_server/ ./model_server/
COPY config.py .

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "model_server.app:app", "--host", "0.0.0.0", "--port", "8000"]
