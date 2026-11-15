FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8081 \
    WHISPER_MODEL=small \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE_TYPE=int8 \
    WHISPER_CPU_THREADS=4 \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HOME=/srv/asr/model_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/asr

COPY asr_service/requirements.txt /srv/asr/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

RUN useradd --create-home --uid 10002 asruser \
    && mkdir -p /srv/asr/model_cache \
    && chown -R asruser:asruser /srv/asr

USER asruser

# Pre-cache CPU int8 Whisper model at build time for sub-2s Cloud Run cold starts
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"

COPY --chown=asruser:asruser asr_service/app /srv/asr/app

EXPOSE 8081

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8081} --proxy-headers"]
