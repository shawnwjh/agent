FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python - <<'PY'
import importlib
for m in ("fastapi","uvicorn"):
    importlib.import_module(m)
print("OK: fastapi/uvicorn installed")
PY

COPY . .

# If your main.py lives under logic/agents/paper-critique-agent/, use --app-dir
CMD uvicorn main:app \
  --app-dir logic/agents/paper-critique-agent \
  --host 0.0.0.0 \
  --port ${PORT:-8080} \
  --log-level debug