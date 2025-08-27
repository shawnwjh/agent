FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    PYTHONFAULTHANDLER=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Gunicorn + Uvicorn worker, bind to $PORT, verbose logs, preload to surface import errors
CMD bash -lc 'exec gunicorn \
  --workers ${WEB_CONCURRENCY:-1} \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind :${PORT:-8080} \
  --access-logfile - --error-logfile - --log-level debug \
  --preload \
  app.main:app'