# Dockerfile (key parts)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python - <<'PY'
import importlib
for m in ("fastapi","uvicorn","gunicorn"):
    importlib.import_module(m)
print("OK: web deps present")
PY

COPY . .

# app/main.py exports `app`, so target app.main:app and bind to $PORT
CMD bash -lc 'exec gunicorn \
  --workers ${WEB_CONCURRENCY:-1} \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind :${PORT:-8080} \
  --access-logfile - --error-logfile - --log-level debug \
  --preload \
  app.main:app'