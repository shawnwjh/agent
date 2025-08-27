FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    PYTHONFAULTHANDLER=1

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Add source
COPY . .

# Start FastAPI on Cloud Run's $PORT with verbose logs
# If your main file is app/main.py and the ASGI variable is `app`, this works:
CMD bash -lc 'exec gunicorn \
  --workers ${WEB_CONCURRENCY:-1} \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind :${PORT:-8080} \
  --access-logfile - --error-logfile - --log-level debug \
  --preload \
  app.main:app'