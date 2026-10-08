FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    RVC_DATASET_DIR=/app/datasets \
    RVC_LOG_DIR=/app/logs \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONIOENCODING=utf-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY tools /app/tools
RUN bash tools/install_fairseq.sh \
    && grep -v '^fairseq==' requirements.txt > /tmp/requirements.nofairseq.txt \
    && pip install --no-cache-dir -r /tmp/requirements.nofairseq.txt \
    && rm /tmp/requirements.nofairseq.txt

COPY . /app
RUN mkdir -p /app/datasets /app/logs /app/assets/weights \
    && chmod +x /app/start.sh

EXPOSE 7860

CMD ["python", "webui.py"]
