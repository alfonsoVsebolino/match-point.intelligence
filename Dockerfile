FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY pipeline/ ./pipeline/
COPY models/ ./models/
COPY data/train_features.csv data/test_features.csv data/match_scores.csv ./data/
COPY web/ ./web/

RUN groupadd --system appuser \
    && useradd --system --gid appuser --create-home appuser \
    && mkdir -p /app/data/predictions_history \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, sys, urllib.request; port = os.environ.get('PORT', '8080'); resp = urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=3); sys.exit(0 if resp.status == 200 else 1)"

CMD ["sh", "-c", "exec uvicorn src.server:app --host 0.0.0.0 --port ${PORT:-8080}"]
