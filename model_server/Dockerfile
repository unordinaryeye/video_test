FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
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
