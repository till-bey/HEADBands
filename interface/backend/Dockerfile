FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PORT=8080

# System libraries required for Piper TTS (espeak-ng + audio I/O)
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    libsndfile1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Copy backend package metadata first for Docker layer caching
COPY backend/pyproject.toml /srv/backend/pyproject.toml
RUN mkdir -p /srv/backend/app && touch /srv/backend/app/__init__.py

WORKDIR /srv/backend
RUN pip install --upgrade pip && pip install .

# Copy backend source and zero-credential static frontend
COPY backend/app /srv/backend/app
COPY frontend /srv/frontend

# Create writable directory for ephemeral runtime Piper ONNX voice cache and non-root user
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /srv/backend/models/piper \
    && chown -R appuser:appuser /srv

USER appuser

# Optionally pre-cache default Leukerbad (Valais / Wallisertiitsch) Piper ONNX voice for instant Cloud Run TTS
ARG PRELOAD_SWISSDIAL_VS=true
RUN if [ "$PRELOAD_SWISSDIAL_VS" = "true" ]; then \
      curl -fsSL "https://huggingface.co/d41k4n/piper-swissdial/resolve/main/de/de_CH/swissdial_vs/medium/de_CH-swissdial_vs-medium.onnx.json" \
        -o /srv/backend/models/piper/de_CH-swissdial_vs-medium.onnx.json && \
      curl -fsSL "https://huggingface.co/d41k4n/piper-swissdial/resolve/main/de/de_CH/swissdial_vs/medium/de_CH-swissdial_vs-medium.onnx" \
        -o /srv/backend/models/piper/de_CH-swissdial_vs-medium.onnx ; \
    fi

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --proxy-headers"]
