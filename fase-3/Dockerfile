FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Código de la Fase 2 y de la API.
COPY pipeline.py train.py inference.py predict.py ./
COPY config.py validation.py service.py api.py ./

ARG GIT_COMMIT=""
ENV GIT_COMMIT=${GIT_COMMIT} \
    MODELS_DIR=models \
    MODEL_VERSION=v1 \
    TRAIN_DATA_PATH=/data/DatosCrudos.csv

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "4", "--timeout", "120", "api:app"]