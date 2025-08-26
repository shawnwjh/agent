FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
CMD bash -lc 'exec gunicorn -k uvicorn.workers.UvicornWorker --bind :${PORT:-8080} main:app'