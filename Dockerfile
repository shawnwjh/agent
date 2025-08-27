FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Start FastAPI on the port Cloud Run provides
CMD bash -lc 'exec gunicorn -k uvicorn.workers.UvicornWorker \
  --bind :${PORT:-8080} app.main:app'